import torch
from FollowbotNets_PPO import FollowbotPolicyNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FollowbotPolicyNet().to(device=device)
net.load_state_dict(torch.load('./saved/Followbot_3_pi_80.pth', map_location="cpu"))

dummy_input = torch.rand((1, 3, 224, 224)).to(device)

torch.onnx.export(net, args=dummy_input, f='./onnx/Followbot_pi.onnx')
