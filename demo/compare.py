import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
# sys.path.insert(0,'/home/gg/body/OSX/OSX/main')
sys.path.insert(0, osp.join('..', 'data'))
# print(os.getwd())
from config import cfg
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='5')
    parser.add_argument('--img_path0', type=str, default='input.png')
    parser.add_argument('--img_path1', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 5, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()

"""for the compare0 input"""
# prepare input image
transform = transforms.ToTensor()
original_img0 = load_img(args.img_path0)
original_img_height, original_img_width = original_img0.shape[:2]
os.makedirs(args.output_folder, exist_ok=True)

# detect human bbox with yolov5s
# detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
detector = torch.hub.load('/home/gg/body/OSX/OSX/yolov5', 'custom', '/home/gg/body/OSX/OSX/yolov5/pt/yolov5s.pt', source='local')

with torch.no_grad():
    results = detector(original_img0)
person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
class_ids, confidences, boxes = [], [], []
for detection in person_results:
    x1, y1, x2, y2, confidence, class_id = detection.tolist()
    class_ids.append(class_id)
    confidences.append(confidence)
    boxes.append([x1, y1, x2 - x1, y2 - y1])
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
vis_img0 = original_img0.copy()

# generate the name for output
input_filename0 = os.path.basename(args.img_path0)
filename0, file_extension0 = os.path.splitext(input_filename0)
bbox_filename0 = f"{filename0}_bbox{file_extension0}"
output_filename0 = f"{filename0}_output{file_extension0}"

for num, indice in enumerate(indices):
    bbox0 = boxes[indice]  # x,y,h,w
    bbox0 = process_bbox(bbox0, original_img_width, original_img_height)
    
    # draw bbox0
    bbox_img0 = original_img0.copy()
    x, y, w, h = bbox0
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(bbox_img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # draw the label
    label = f"Person {num + 1}: {class_ids[indice]} ({confidences[indice]:.2f})"
    cv2.putText(bbox_img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img0, bbox0, 1.0, 0.0, False, cfg.input_img_shape)
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    # mesh recovery
    with torch.no_grad():
        out0 = demoer.model(inputs, targets, meta_info, 'test')

"""for the compare1 input"""
# prepare input image
original_img1 = load_img(args.img_path1)
original_img_height, original_img_width = original_img1.shape[:2]
os.makedirs(args.output_folder, exist_ok=True)

with torch.no_grad():
    results = detector(original_img1)
person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
class_ids, confidences, boxes = [], [], []
for detection in person_results:
    x1, y1, x2, y2, confidence, class_id = detection.tolist()
    class_ids.append(class_id)
    confidences.append(confidence)
    boxes.append([x1, y1, x2 - x1, y2 - y1])
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
vis_img1 = original_img1.copy()

# generate the name for output
input_filename1 = os.path.basename(args.img_path1)
filename1, file_extension1 = os.path.splitext(input_filename1)
bbox_filename1 = f"{filename1}_bbox{file_extension1}"
output_filename1 = f"{filename1}_output{file_extension1}"

for num, indice in enumerate(indices):
    bbox1 = boxes[indice]  # x,y,h,w
    bbox1 = process_bbox(bbox1, original_img_width, original_img_height)
    
    # draw bbox1
    bbox_img1 = original_img1.copy()
    x, y, w, h = bbox1
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(bbox_img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # draw the label
    label = f"Person {num + 1}: {class_ids[indice]} ({confidences[indice]:.2f})"
    cv2.putText(bbox_img1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img1, bbox1, 1.0, 0.0, False, cfg.input_img_shape)
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    # mesh recovery
    with torch.no_grad():
        out1 = demoer.model(inputs, targets, meta_info, 'test')

# save the bbox
cv2.imwrite(os.path.join(args.output_folder, bbox_filename0), bbox_img0[:, :, ::-1])
cv2.imwrite(os.path.join(args.output_folder, bbox_filename1), bbox_img1[:, :, ::-1])

# generate mesh0
mesh0 = out0['smplx_mesh_cam'].detach().cpu().numpy()
mesh0 = mesh0[0]
# save mesh
# save_obj(mesh0, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))
# render mesh0
focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox0[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox0[3]]
princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox0[2] + bbox0[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox0[3] + bbox0[1]]
vis_img0 = render_mesh(vis_img0, mesh0, smpl_x.face, {'focal': focal, 'princpt': princpt})


# generate mesh1
mesh1 = out1['smplx_mesh_cam'].detach().cpu().numpy()
mesh1 = mesh1[0]
# render mesh1
focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox1[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox1[3]]
princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox1[2] + bbox1[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox1[3] + bbox1[1]]
vis_img1 = render_mesh(vis_img1, mesh1, smpl_x.face, {'focal': focal, 'princpt': princpt})

# save rendered image
cv2.imwrite(os.path.join(args.output_folder, output_filename0), vis_img0[:, :, ::-1])
cv2.imwrite(os.path.join(args.output_folder, output_filename1), vis_img1[:, :, ::-1])

import torch.nn.functional as F
# use body_pose which is from body_regressor.
compare_body_pose = F.cosine_similarity(out0['smplx_body_pose'], out1['smplx_body_pose']).item()
def angle_sim(aa0,aa1):
    c0 = aa0.reshape(21,3)
    c1 = aa1.reshape(21,3)
    c0_sums = torch.sum(torch.pow(c0, 2), dim=1)
    c1_sums = torch.sum(torch.pow(c1, 2), dim=1)
    c0 = torch.sqrt(c0_sums)
    c1 = torch.sqrt(c1_sums)
    add = (torch.abs(c1)+torch.abs(c0))/2
    dif = torch.mean(torch.abs((c1-c0)/add))
    a_sim = 1/(1+pow(dif,2))
    return a_sim
ggcompare_body_pose = compare_body_pose*angle_sim(out0['smplx_body_pose'], out1['smplx_body_pose']).item()

compare_body_pose_quaternion = F.cosine_similarity(out0['body_quaternion'], out1['body_quaternion']).item()
compare_body_pose_euler = F.cosine_similarity(out0['body_ea'], out1['body_ea']).item()

# use body_pose_token which if from encoder.
compare_body_pose_token = F.cosine_similarity(out0['body_pose_token'].view(-1).unsqueeze(0), out1['body_pose_token'].view(-1).unsqueeze(0)).item()
compare_new_body_pose_token = F.cosine_similarity(out0['new_body_pose_token'].view(-1).unsqueeze(0), out1['new_body_pose_token'].view(-1).unsqueeze(0)).item()
compare_body_pose_rot6d = F.cosine_similarity(out0['body_pose_rot6d'], out1['body_pose_rot6d']).item()
compare_body_pose_rotmat = F.cosine_similarity(out0['body_pose_rotmat'], out1['body_pose_rotmat']).item()

euclidean_distance = torch.norm(out0['smplx_body_pose']- out1['smplx_body_pose']).item()
euclidean_compare = 1.0/(1.0+euclidean_distance ** 2)

def manhattan_d(vector_a, vector_b):
    distance = torch.sum(torch.abs(vector_a - vector_b))
    return distance.item()
manhattan_distance = manhattan_d(out0['smplx_body_pose'], out1['smplx_body_pose'])
manhattan_compare = 1.0/(1.0+manhattan_distance ** 2)

# add file
def append_to_file(file_path, content):
    try:
        # 打开文件，尝试读取内容
        with open(file_path, 'r') as file:
            existing_content = file.read()
    except FileNotFoundError:
        # 文件不存在，创建文件并写入内容
        with open(file_path, 'w') as file:
            file.write(content)
    else:
        # 文件存在且不为空，追加内容
        with open(file_path, 'a') as file:
            file.write(content)

file_path = os.path.join(args.output_folder, 'score_output.txt')
# content_to_append = input_filename0+' vs '+input_filename1+'\n'+"cosine score: "+str(compare)+'\n'+"euclidean score: "+str(euclidean_compare)+'\n'+'\n'
content_to_append = input_filename0+' vs '+input_filename1+'\n'+ \
    "compare_body_pose cosine score: "+str(compare_body_pose)+'\n'+ \
        "compare_body_pose ggcosine score: "+str(ggcompare_body_pose)+'\n'+ \
        "compare_body_pose_quaternion cosine score: "+str(compare_body_pose_quaternion)+'\n'+ \
        "compare_body_pose_euler cosine score: "+str(compare_body_pose_euler)+'\n'+ \
        "compare_body_pose_token cosine score: "+str(compare_body_pose_token)+'\n'+ \
        "compare_new_body_pose_token cosine score: "+str(compare_new_body_pose_token)+'\n'+ \
        "compare_body_pose_rot6d cosine score: "+str(compare_body_pose_rot6d)+'\n'+ \
        "compare_body_pose_rotmat cosine score: "+str(compare_body_pose_rotmat)+'\n'+'\n'
append_to_file(file_path, content_to_append)

def angle_dif(aa0,aa1):
    c0 = aa0.reshape(21,3)
    c1 = aa1.reshape(21,3)
    c0_sums = torch.sum(torch.pow(c0, 2), dim=1)
    c1_sums = torch.sum(torch.pow(c1, 2), dim=1)
    c0 = torch.sqrt(c0_sums)
    c1 = torch.sqrt(c1_sums)
    add = (torch.abs(c1)+torch.abs(c0))/2
    dif = torch.abs((c1-c0)/add)
    return dif

save_obj(mesh0, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))