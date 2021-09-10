import onnx
import onnxruntime
import torch
from model.models import YOLOV4
import argparse
from model.models import parseCfgFile
import numpy as np

#验证onnx模型和pytorch模型损失精度的脚本

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
if __name__ == '__main__':

    project_path = 'F:/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,default=r'F:/YOLOv4_project_ylc20210526/cfg/yolov5-0721-s-s-up-v4-pro-eightrep-csp9-all_test_new_spp.cfg',
                        help='配置文件')
    parser.add_argument('--onnx_save_path', type=str,default='C:/Users/ylcServer/Desktop/0827/yolov4_0827_v11.onnx',
                        help='标签地址')
    parser.add_argument('--checkpoint', type=str,default=r'C:/Users/ylcServer/Desktop/0827/YOLOv4_project_ylc20210526_ckpt_1200.pth',
                        help='存档地址')

    opt = parser.parse_args()
    cfgs = parseCfgFile(opt.cfg)

    # 读取net部分配置
    net_cfg = cfgs[0]

    # 图片大小
    image_size = net_cfg['width']
    # 图片通道数
    channels = net_cfg['channels']

    # gpu设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu")

    #img=np.random.randn(1,1,1024,1024).astype(dtype=np.float32)
    img=torch.randn(1,1,1024,1024).to(device)

    net = YOLOV4(cfg=cfgs[1:],  channels=channels).to(device)
    model = torch.load(opt.checkpoint)
    net.load_state_dict(model)
    net.eval()
    net.fuse(is_rep=True)
    #net.half()


    # def onnx_infer(model_path, img_data):
    #     onnx.checker.check_model(onnx.load(model_path))
    #     sess = onnxruntime.InferenceSession(model_path)
    #     out_names = ["output"]
    #     results = sess.run(out_names, {"input": img_data})
    #
    #     # results是list，里面装的是numpy.ndarray
    #     return results[0]
    #
    # result=onnx_infer(opt.onnx_save_path,img)
    # print(result)
    # exit()

    #
    #onnx_model=onnx.load(opt.onnx_save_path).to(device)

    # 通过检查模型的版本，图的结构以及节点及其输入和输出，可以验证 ONNX 图的有效性
    onnx.checker.check_model(onnx.load(opt.onnx_save_path))
    # 验证模型是否匹配
    sess=onnxruntime.InferenceSession(opt.onnx_save_path)
    out_name=["output"]
    np_img =  to_numpy(img)#.astype(np.float32)
    onnx_out = sess.run(out_name, {"input": np_img})

    print("onnxout", onnx_out)
    #exit()
    with torch.no_grad():
        #img=img.half()
        pt_out=net(img)

        print("ptout",pt_out)

    onnx_ou2=torch.tensor(onnx_out[0])

    print(np.abs(onnx_out[0]-to_numpy(pt_out[0])).mean())
    print(np.abs(onnx_out[0]-to_numpy(pt_out[0])).max())
    # print((onnx_ou2 - pt_out[0]).min())

    # 单元测试通常使用断言函数作为测试的组成部分
    # assert_almost_equal 如果两个数字的近似程度没有达到指定精度，就抛出异常, decimal为指定精度位数,小数点后3位
    np.testing.assert_almost_equal(to_numpy( pt_out[0]), onnx_out[0], decimal=3)

    # cp = onnx.helper.printable_graph(onnx.load(opt.onnx_save_path).graph)
    # print(cp)

