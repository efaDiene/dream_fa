import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import trimesh
import rembg

from cam_utils import orbit_camera, extract_azimuth_elevation, OrbitCamera
from mesh_renderer import Renderer

# from kiui.lpips import LPIPS

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(opt).to(self.device)

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input image2
        self.input_img2 = None
        self.input_mask2 = None
        self.input_img_torch2 = None
        self.input_mask_torch2 = None
        self.overlay_input_img2 = False
        self.overlay_input_img_ratio2 = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        # self.lpips_loss = LPIPS(net='vgg').to(self.device)
        
        self.T1 = self.T2 = self.T3 = self.T4 = np.eye(4)

        T1=np.array([[0.9958024195334342,-0.018871573594701076,0.0895622965394354,1.0876262766729021],
                    [0.09136257244450904,0.1459772386914313,-0.9850601637160781,-4.289772970297661],
                    [0.0055155786350863335,0.9891079362203522,0.1470886429954247,0.6471367509962119],
                    [0.0, 0.0,0.0, 1.0]])

        T2=np.array([[-0.7255133758153133, 0.16008781379961542,-0.669329689604404,  -3.5211266181713743],
                    [-0.6882000612020779,-0.16407294163500233,0.7067253678655477, 2.7937770033229232 ],
                    [ 0.003319228001286951, 0.9733714407645709, 0.2292095570201615, 0.6019382195158107 ], 
                    [0.0, 0.0, 0.0,1.0]]
)
        T3=np.array([[-0.6746754311116187,-0.08681601004864213,0.7329911616476597,3.2015423336685735 ],
                    [0.736727706605517,-0.14005170023378533, 0.6615268759327932,2.5178358588686636],
                    [0.04522553453666382, 0.9863308276946736, 0.15844920121292183,0.6379604279301839 ],
                    [ 0.0,0.0,0.0,1.0]])
        T4=np.array([[0.3922094416962398,0.20701126227606687,-0.8962801409911931,-4.235969771002857],
                    [-0.9196552365322843,0.1095852900501236,-0.37712771062781675,-1.7791805239229939],
                    [0.020149435800337998,0.972181773895962,0.23335937680202273,0.5725921638470095],
                    [ 0.0,0.0, 0.0,1.0]])
    
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input,T1)
        # load input data from cmdline
        if self.opt.input2 is not None:
            self.load_input2(self.opt.input2,T2)
        # load input data from cmdline
        if self.opt.input3 is not None:
            self.load_input3(self.opt.input3,T3)
        # load input data from cmdline
        if self.opt.input4 is not None:
            self.load_input4(self.opt.input4,T4)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def padding(self, img):
        h,w = img.shape[:2]
        if h != w:
            max_edge = max(h,w)
            x_center = (max_edge - w) // 2
            y_center = (max_edge - h) // 2


            if len(img.shape) == 3:
                c = img.shape[2]
                padding_img = np.ones((max_edge,max_edge,c))
                padding_img[y_center:y_center + h, x_center:x_center + w, :] = img
            else:
                padding_img = np.ones((max_edge,max_edge))
                padding_img[y_center:y_center+h, x_center:x_center+w] = img
            img = padding_img
        return img

    def generate_box(self, input_mask):
        # generate bbox
        # input_mask = img[..., 3:]
        rows = np.any(input_mask, axis=1)
        cols = np.any(input_mask, axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Create the bounding box (top-left and bottom-right coordinates)
        bbox = [col_min, row_min, col_max, row_max]

        return bbox
    
    def recenter2(self, img, bbox, width, height):
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        img_center_x, img_center_y = width / 2, height / 2
        shift_x = img_center_x - bbox_center_x
        shift_y = img_center_y - bbox_center_y
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (width, height))
        return img
    
    def recenter(self, img, bbox, width, height,transform_matrix, fx=1363.3857, fy=1358.9104):
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        img_center_x, img_center_y = width / 2, height / 2
        shift_x = img_center_x - bbox_center_x
        shift_y = img_center_y - bbox_center_y
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (width, height))
        # Conversion en 3D
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        delta_camera = np.array([shift_x * np.linalg.norm(t) / fx, shift_y * np.linalg.norm(t) / fy, 0.0])
        delta_world = np.dot(R, delta_camera)
        t_new=t+delta_world
        transform_matrix[:3, 3] = t_new
        return img , transform_matrix


    def prepare_train(self):

        self.step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            T1=np.array([[0.9958024195334342,-0.018871573594701076,0.0895622965394354,1.0876262766729021],
                        [0.09136257244450904,0.1459772386914313,-0.9850601637160781,-4.289772970297661],
                        [0.0055155786350863335,0.9891079362203522,0.1470886429954247,0.6471367509962119],
                        [0.0, 0.0,0.0, 1.0]])

            T2=np.array([[-0.7255133758153133, 0.16008781379961542,-0.669329689604404,  -3.5211266181713743],
                        [-0.6882000612020779,-0.16407294163500233,0.7067253678655477, 2.7937770033229232 ],
                        [ 0.003319228001286951, 0.9733714407645709, 0.2292095570201615, 0.6019382195158107 ], 
                        [0.0, 0.0, 0.0,1.0]]
  )
            T3=np.array([[-0.6746754311116187,-0.08681601004864213,0.7329911616476597,3.2015423336685735 ],
                        [0.736727706605517,-0.14005170023378533, 0.6615268759327932,2.5178358588686636],
                        [0.04522553453666382, 0.9863308276946736, 0.15844920121292183,0.6379604279301839 ],
                        [ 0.0,0.0,0.0,1.0]])
            T4=np.array([[0.3922094416962398,0.20701126227606687,-0.8962801409911931,-4.235969771002857],
                        [-0.9196552365322843,0.1095852900501236,-0.37712771062781675,-1.7791805239229939],
                        [0.020149435800337998,0.972181773895962,0.23335937680202273,0.5725921638470095],
                        [ 0.0,0.0, 0.0,1.0]])
            azimuth1, elevation1,ti=extract_azimuth_elevation(T1)
            azimuth2, elevation2,ti=extract_azimuth_elevation(T2)
            azimuth3, elevation3,ti=extract_azimuth_elevation(T3)
            azimuth4, elevation4,ti=extract_azimuth_elevation(T4)
            #pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            pose = orbit_camera( elevation1, azimuth1 , self.opt.radius)
            pose2 = orbit_camera(elevation2, azimuth2 , self.opt.radius)
            pose3 = orbit_camera(elevation3, azimuth3 , self.opt.radius)
            pose4 = orbit_camera(elevation4, azimuth4 , self.opt.radius)  
        self.fixed_cam = (pose, self.cam.perspective)
        self.fixed_cam2 = (pose2, self.cam.perspective)
        self.fixed_cam3 = (pose3, self.cam.perspective)
        self.fixed_cam4 = (pose4, self.cam.perspective)
        
        

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")
        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()

            #input2
        if self.opt.input2 is not None:
            self.input_img_torch2 = torch.from_numpy(self.input_img2).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch2 = F.interpolate(self.input_img_torch2, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch2 = torch.from_numpy(self.input_mask2).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch2 = F.interpolate(self.input_mask_torch2, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last2 = self.input_img_torch2[0].permute(1,2,0).contiguous()

            #input3
        if self.opt.input3 is not None:
            self.input_img_torch3 = torch.from_numpy(self.input_img3).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch3 = F.interpolate(self.input_img_torch3, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch3 = torch.from_numpy(self.input_mask3).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch3 = F.interpolate(self.input_mask_torch3, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last3 = self.input_img_torch3[0].permute(1,2,0).contiguous()

            #input4
        if self.opt.input4 is not None:
            self.input_img_torch4 = torch.from_numpy(self.input_img4).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch4 = F.interpolate(self.input_img_torch4, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch4 = torch.from_numpy(self.input_mask4).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch4 = F.interpolate(self.input_mask_torch4, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last4 = self.input_img_torch4[0].permute(1,2,0).contiguous()



        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()


        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:

                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(*self.fixed_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)
                out2 = self.renderer.render(*self.fixed_cam2, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)
                out3 = self.renderer.render(*self.fixed_cam3, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)
                out4 = self.renderer.render(*self.fixed_cam4, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)


                # rgb loss
                image = out["image"] # [H, W, 3] in [0, 1]
                valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()
                loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)

                
            if self.opt.input2 is not None:
                # rgb loss
                image2 = out2["image"] # [H, W, 3] in [0, 1]
                valid_mask2 = ((out2["alpha"] > 0) & (out2["viewcos"] > 0.5)).detach()
                loss = loss + F.mse_loss(image2 * valid_mask2, self.input_img_torch_channel_last2 * valid_mask2)

            if self.opt.input3 is not None:
                # rgb loss
                image3 = out3["image"] # [H, W, 3] in [0, 1]
                valid_mask3 = ((out3["alpha"] > 0) & (out3["viewcos"] > 0.5)).detach()
                loss = loss + F.mse_loss(image3 * valid_mask3, self.input_img_torch_channel_last3 * valid_mask3)

            if self.opt.input4 is not None:
                # rgb loss
                image4 = out4["image"] # [H, W, 3] in [0, 1]
                valid_mask4 = ((out4["alpha"] > 0) & (out4["viewcos"] > 0.5)).detach()
                loss = loss + F.mse_loss(image4 * valid_mask4, self.input_img_torch_channel_last4 * valid_mask4)

            ### novel view (manual batch)
            render_resolution = 512
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                images.append(image)
                image2 = image.squeeze(0)  # Passer de [1, 3, H, W] à [3, H, W]
                image2 = image2.permute(1, 2, 0)  # Passer de [3, H, W] à [H, W, 3]

                # Assurez-vous que les valeurs sont entre 0 et 255 (pour l'image RGB)
                image2 = (image2 * 255).clamp(0, 255).byte()  # Normaliser et convertir en entier

                # Enregistrer l'image avec OpenCV
                cv2.imwrite('output_image_opencv3.jpg', image2.cpu().numpy())

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                        image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # import kiui
            # kiui.lo(hor, ver)
            # kiui.vis.plot_image(image)

            # guidance loss
            strength = step_ratio * 0.15 + 0.8
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                    refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
                else:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                    refined_images = self.guidance_sd.refine(images, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)

            if self.enable_zero123:
                # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
                # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()
            
            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    def load_input4(self, file4,T):
        # load image
        print(f'[INFO] load image from {file4}...')
        img4 = cv2.imread(file4, cv2.IMREAD_UNCHANGED)
        if img4.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img4 = rembg.remove(img4, session=self.bg_remover)

        #img4 = cv2.resize(img4, (self.W, self.H), interpolation=cv2.INTER_AREA)
        #img4 = self.padding(img4)
        img4 = img4.astype(np.float32) / 255.0

        #self.input_mask4 = img4[..., 3:]
        mask = img4[..., -1] > 0
        # white bg
        bbox = self.generate_box(mask)
        height, width = img4.shape[:2]
        carved_image, self.T4 = self.recenter(img4, bbox, width, height,T)
        #carved_image = self.padding(carved_image)
        
        self.input_mask4 = carved_image[..., 3:]
        # white bg
        self.input_img4 = carved_image[..., :3] * self.input_mask4 + (1 - self.input_mask4)
        # bgr to rgb
        self.input_img4 = self.input_img4[..., ::-1].copy()


    def load_input3(self, file3,T):
        # load image
        print(f'[INFO] load image from {file3}...')
        img3 = cv2.imread(file3, cv2.IMREAD_UNCHANGED)
        if img3.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img3 = rembg.remove(img3, session=self.bg_remover)

        #img3 = cv2.resize(img3, (self.W, self.H), interpolation=cv2.INTER_AREA)
        #img3 = self.padding(img3)
        img3 = img3.astype(np.float32) / 255.0

        #self.input_mask3 = img3[..., 3:]
        mask = img3[..., -1] > 0
        # white bg
        bbox = self.generate_box(mask)
        height, width = img3.shape[:2]
        carved_image,self.T3 = self.recenter(img3, bbox, width, height,T)
        #carved_image = self.padding(carved_image)

        
        self.input_mask3 = carved_image[..., 3:]
        # white bg
        self.input_img3 = carved_image[..., :3] * self.input_mask3 + (1 - self.input_mask3)
        # bgr to rgb
        self.input_img3 = self.input_img3[..., ::-1].copy()


    def load_input2(self, file2,T):
        # load image
        print(f'[INFO] load image from {file2}...')
        img2 = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
        if img2.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img2 = rembg.remove(img2, session=self.bg_remover)

        #img2 = cv2.resize(img2, (self.W, self.H), interpolation=cv2.INTER_AREA)
        #img2 = self.padding(img2)
        img2 = img2.astype(np.float32) / 255.0

        #self.input_mask2 = img2[..., 3:]
        mask = img2[..., -1] > 0
        # white bg
        bbox = self.generate_box(mask)
        height, width = img2.shape[:2]
        carved_image,self.T2 = self.recenter(img2, bbox, width, height,T)
        #carved_image = self.padding(carved_image)
        
        self.input_mask2 = carved_image[..., 3:]
        # white bg
        self.input_img2 = carved_image[..., :3] * self.input_mask2 + (1 - self.input_mask2)
        # bgr to rgb
        self.input_img2 = self.input_img2[..., ::-1].copy()

    def load_input(self, file,T):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        #img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        #img = self.padding(img)
        img = img.astype(np.float32) / 255.0

        #self.input_mask = img[..., 3:]
        mask = img[..., -1] > 0
        # white bg
        bbox = self.generate_box(mask)
        height, width = img.shape[:2]
        carved_image,self.T1 = self.recenter(img, bbox, width, height,T)
        #carved_image = self.padding(carved_image)

        self.input_mask = carved_image[..., 3:]
        # white bg
        self.input_img = carved_image[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
        # save
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)
