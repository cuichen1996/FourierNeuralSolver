import numpy as np
import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, config):
        super(DeepONet, self).__init__()
        self.p = config["p"]
        self.relu_activation = nn.ReLU()
        self.tanh_activation = nn.Tanh()
        self.num_of_inputs = config["input_channel"] # channel of PDE paras


        #* residual branch: linear on bias
        num_of_conv_layers = 5
        layers_list = [nn.Conv2d(in_channels=1,
                                                out_channels=20 + 20,
                                                kernel_size=(3, 3),
                                                stride=2,
                                                padding='valid', bias=False)]
        for conv_layer_index in range(2, num_of_conv_layers + 1):
            layers_list.append(nn.Conv2d(in_channels=20 * (2 ** (conv_layer_index - 2)) + 20,
                                            out_channels=20 * (2 ** (conv_layer_index - 1)) + 20,
                                            kernel_size=(3, 3),
                                            stride=2,
                                            padding='valid',
                                            bias=False))
            
        layers_list.append(nn.AdaptiveAvgPool2d(3))
        layers_list.append(nn.Flatten())
        layers_list.append(nn.Linear((20 * (2 ** (num_of_conv_layers - 1)) + 20)*9, 80))
        layers_list.append(nn.Linear(80, 80))
        layers_list.append(nn.Linear(80, self.p))

        self.branch_r = nn.Sequential(*layers_list)


        #* PDE paras branch: nonlinear
        layers_list = [nn.Conv2d(in_channels=self.num_of_inputs,
                                                out_channels=20 + 20,
                                                kernel_size=(3, 3),
                                                stride=2,
                                                padding='valid'),
                        nn.ReLU()]
        for conv_layer_index in range(2, num_of_conv_layers + 1):
            layers_list.append(nn.Conv2d(in_channels=20 * (2 ** (conv_layer_index - 2)) + 20,
                                            out_channels=20 * (2 ** (conv_layer_index - 1)) + 20,
                                            kernel_size=(3, 3),
                                            stride=2,
                                            padding='valid'))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.AdaptiveAvgPool2d(3))
        layers_list.append(nn.Flatten())
        layers_list.append(nn.Linear((20 * (2 ** (num_of_conv_layers - 1)) + 20)*9, 80))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(80, 80))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(80, self.p))

        self.branch_p = nn.Sequential(*layers_list)

        self.bias = nn.Parameter(torch.tensor(0.0, requires_grad=True,
                                              dtype=torch.float))
        # Trunk Net
        self.trunk_net = nn.Sequential(nn.Linear(2, 80),
                                       self.tanh_activation,
                                       nn.Linear(80, 80),
                                       self.tanh_activation,
                                       nn.Linear(80, self.p),
                                       self.tanh_activation)

    def _forward_branch(self, branch_input_k, branch_input_f):
        out1 = self.branch_p(branch_input_k)
        out2 = self.branch_r(branch_input_f)
        return out1*out2

    def generate_trunk_inputs(self, N, inputs=None):
        if inputs is None:
            x_nodes = np.linspace(0,1,N)[1:-1]
            y_nodes = np.linspace(0,1,N)[1:-1]
            xv, yv = np.meshgrid(x_nodes, y_nodes, indexing='ij')
            xv_flat, yv_flat = xv.flatten(), yv.flatten()
            return torch.tensor(np.stack([xv_flat, yv_flat], axis=1),
                                requires_grad=True,
                                dtype=torch.float)

        else:
            return torch.tensor(inputs[:, None], requires_grad=True,
                                dtype=torch.float)

    def forward(self, branch_input_k, branch_input_f, trunk_inputs=None):
        N = branch_input_k.shape[-1] + 1
        if trunk_inputs is None:
            trunk_inputs = self.generate_trunk_inputs(N)
            is_internal = True
        else:
            trunk_inputs = self.generate_trunk_inputs(N, trunk_inputs)
            is_internal = False
        trunk_outputs = self.trunk_net(trunk_inputs.to(branch_input_k.device))       # (num_points, self.p)

        # f(x), or the RHS: make L2 norm = 1, and save the magnitude (multiply back later)
        f_norm = torch.sqrt(torch.mean(branch_input_f ** 2, dim=(2, 3), keepdim=True))
        branch_input_f = branch_input_f / f_norm

        branch_outputs = self._forward_branch(branch_input_k, branch_input_f)

        # When training
        if is_internal:
            u_pred = torch.reshape(branch_outputs @ trunk_outputs.T,
                                    (branch_outputs.shape[0], 1,
                                    N-2,
                                    N-2))

        else: # When inferring
            u_pred = torch.reshape(branch_outputs @ trunk_outputs.T[:, 0, :],
                                    (branch_outputs.shape[0], 1,
                                    N-2,
                                    N-2))
        u_pred = u_pred * f_norm
        return u_pred