import onnx
import torch
from model.models import YOLOV4
import argparse
from model.models import parseCfgFile
import torch._C as _C
TrainingMode = _C._onnx.TrainingMode

"""
Open Neural Network Exchange（ONNX，开放神经网络交换）格式，是一个用于表示深度学习模型的标准，可使模型在不同框架之间进行转移。

有两种方式可用于保存/加载pytorch模型 
1）文件中保存模型结构和权重参数 存储:torch.save(model,"model_name.pth")  加载:model=torch.load("model_name.pth")
2）文件只保留模型权重.  存储:torch.save(model.state_dict(),"model_name.pth")  加载:model.load.state_dict(torch.load("model_name.pth")
"""
#pytorch模型转onnx脚本

def get_onnx(model, onnx_save_path, example_tensor):

    #example_tensor = example_tensor.cuda()
    #example_tensor = example_tensor
    """
    torch.onnx.export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None)
    model(torch.nn.Module)-要被导出的模型
    args(参数的集合)-模型的输入
    f一个类文件的对象或一个包含文件名字符串
    export_params(bool,default True)-如果指定，所有参数都会被导出。如果你只想导出一个未训练的模型，就将此参数设置为False
    verbose(bool,default False)-如果指定，将会输出被导出的轨迹的调试描述
    training(bool,default False)-导出训练模型下的模型。目前，ONNX只面向推断模型的导出，所以一般不需要将该项设置为True。
    input_names(list of strings, default empty list)-按顺序分配名称到图中的输入节点
    output_names(list of strings, default empty list)-按顺序分配名称到图中的输出节点
    """
    _ = torch.onnx.export(model,  # model being run
                          example_tensor,  # model input (or a tuple for multiple inputs)
                          onnx_save_path,
                          verbose=True,  # store the trained parameter weights inside the model file
                          input_names=['input'],
                          output_names=['output'],
                          opset_version=11
                          )

if __name__ == '__main__':

    project_path = 'F:/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,default=r'F:/YOLOv4_project_ylc20210526/cfg/yolov5-0721-s-s-up-v4-pro-eightrep-csp9-all_test_new_spp.cfg',
                        help='配置文件')
    parser.add_argument('--onnx_save_path', type=str,default=project_path + 'YOLOv4_project_ylc20210526/onnx_pt/yolov4_0827_v11.onnx',
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

    #example_tensor = torch.randn(1, 1, 1024, 1024, device="cpu")
    example_tensor = torch.randn(1, 1, 1024, 1024).to(device)  # 示例输入
    #example_tensor = example_tensor.half()

    # 保存模型权重的方式
    net = YOLOv4(cfg=cfgs[1:],  channels=channels).to(device)  # 初始化模型
    model = torch.load(opt.checkpoint)  # 加载模型
    net.load_state_dict(model)  # 赋值模型权重

    # 保存模型结构和模型权重的方式
    #net=torch.load(opt.checkpoint)

    net.eval()  # 设置模型处于推理模式
    net.fuse(is_rep=True)

    #net.half()
    # 导出模型
    get_onnx(net, opt.onnx_save_path, example_tensor)