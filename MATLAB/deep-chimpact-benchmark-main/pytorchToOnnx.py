# In order to run this code, we need to change
# networks/depth_decoder.py Line 63 to:
# self.outputs[("disp", str(i))] = self.sigmoid(self.convs[("dispconv", i)](x))

import torch
import networks

encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load("encoder.pth")
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load('depth.pth')
depth_decoder.load_state_dict(loaded_dict)

dummy_input_enc = torch.randn(1,3,192,640)
dummy_input_dec = [torch.randn(1,64,96,320), torch.randn(1,64,48,160), torch.randn(1,128,24,80), torch.randn(1,256,12,40),torch.randn(1,512,6,20)] 

torch.onnx.export(encoder, dummy_input_enc, "onnx_encoder.onnx")
torch.onnx.export(depth_decoder, dummy_input_dec, "onnx_decoder.onnx",opset_version=11)

