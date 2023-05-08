import torch
import time

assert torch.cuda.device_count() >= 2


# pytorch 可以实现在不同设备上的自动并行计算
# class Benchmark:
#     def __init__(self, prefix=None):
#         self.prefix = prefix + ' ' if prefix else ''
#
#     def __enter__(self):
#         self.start = time.time()
#
#     def __exit__(self, *args):
#         print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))
#
#
# def run(x):
#     for _ in range(10000):
#         y = torch.mm(x, x)
#
#
# x_gpu1 = torch.rand((100, 100), device='cuda:0')
# x_gpu2 = torch.rand((100, 100), device='cuda:1')
#
# with Benchmark('Run on GPU1.'):
#     run(x_gpu1)
#     torch.cuda.synchronize()
#
# with Benchmark('Run on GPU1.'):
#     run(x_gpu1)
#     torch.cuda.synchronize()
#
# with Benchmark('Run on both GPU1 and GPU2 in parallel.'):
#     run(x_gpu1)
#     run(x_gpu2)
#     torch.cuda.synchronize()

net = torch.nn.Linear(10, 1).cuda()
net = torch.nn.DataParallel(net, device_ids=[0, 3]) # 指定哪几个gpu工作

# 多GPU模型的保存与加载
torch.save(net.module.state_dict(), "./model.pt")

new_net = torch.nn.Linear(10, 1)
new_net.load_state_dict(torch.load('./model.pt'))


