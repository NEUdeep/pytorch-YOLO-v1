#encoding:utf-8
#
#created by xiongzihua
#
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

def decoder(pred,grid_num):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes=[] # 用来放框，和下面是对应关系,取列就是一个对象的box，label，score了，真的很方面，很清晰啊；
    cls_indexs=[] # 用来放有对象的label
    probs = [] # 用来放有对象的box的score
    cell_size = 1./grid_num # ？
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2) #torch.Size([7, 7, 1]);取了第一个box的net预测的confidence
    contain2 = pred[:,:,9].unsqueeze(2) #torch.Size([7, 7, 1]);取了第2个box的net预测的confidence
    contain = torch.cat((contain1,contain2),2) #torch.Size([7, 7, 2]);
    mask1 = contain > 0.1 #大于阈值 ,是bool值,这个0.1过滤掉了90%不合格的box，按道理来讲，这一步是nms来做的，即设定一个阈值，直接过掉低score于该阈值的框
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0) # 对求和以后的mask进行逐元素和0比较，大于0则为true，似乎在这里没啥意义
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1: # 对符合大于0.1的置信度的框进行操作
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4] # 找到该grid对应的框，乘5的意义是两个框间隔5个元素连续存放
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]]) # 对应box的confidence
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image ；tensor([0.3689, 0.5461, 0.2675, 0.5443])
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,y1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:] #tensor([0.2352, 0.2740, 0.5027, 0.8183])
                    max_prob,cls_index = torch.max(pred[i,j,10:],0) # 得到该grid最大的概率和对应的label
                    if float((contain_prob*max_prob)[0]) > 0.1: # contain_prob*max_prob就是该grid的score
                        boxes.append(box_xy.view(1,4)) #[tensor([[0.2352, 0.2740, 0.5027, 0.8183]])]
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0: #boxes:[tensor([[0.2352, 0.2740, 0.5027, 0.8183]]), tensor([[0.2563, 0.2567, 0.4228, 0.8374]]), tensor([[0.6202, 0.2711, 0.9388, 0.7943]]), tensor([[0.2039, 0.3376, 0.4831, 0.8453]]), tensor([[0.6671, 0.3296, 0.8981, 0.8378]]), tensor([[0.0358, 0.6046, 0.3456, 0.8906]])]
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        new_clsindex = [torch.tensor([x]) for x in cls_indexs] # it was be ok! 
        cls_indexs = torch.cat(new_clsindex,0) #(n,) # when you got a tensor(6) not tensor([6]), it will throw RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated in pytorch>0.3;
    keep = nms(boxes,probs) # 返回的是nms以后留下框的index
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5): # 阈值越低，框recall的越少。
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0] #tensor([0.2352, 0.2563, 0.6202, 0.2039, 0.6671, 0.0358])
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1) # 得到每个框的面积；tensor([0.1456, 0.0967, 0.1667, 0.1417, 0.1174, 0.0886])

    _,order = scores.sort(0,descending=True) # 对score进行排序，目的是先输出最大得分的框为标准，然后让其他的框和它算IOU，即交并比；
    keep = []
    # import pdb
    # pdb.set_trace()
    while order.numel() > 0:
        if order.numel() == 1:
            i = order
        else:
            i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i]) # x1[order[1:]]：tensor([0.2563, 0.6202, 0.2039, 0.6671, 0.0358])；clamp(min=x1[i])是将最后一个元素替换为x1[i];tensor([0.2563, 0.6202, 0.2352, 0.6671, 0.2352])；
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h #交的面积

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze() #tensor([1, 3, 4])
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name,grid_num,root_path=''):

    result = []
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:],volatile=True)
    img = img.cuda()

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred,grid_num)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result
        



if __name__ == '__main__':
    grid_num = 7
    if grid_num == 7:
        model = vgg16_bn()
    else:
        model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('checkpoint/map0.53_vgg16_bn_160_best.pth'))
    model.eval()
    model.cuda()
    image_name = '2.jpeg'
    image = cv2.imread(image_name)
    print('predicting...')
    result = predict_gpu(model,image_name,grid_num)
    for left_up,right_bottom,class_name,_,prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)
    print('Done')



