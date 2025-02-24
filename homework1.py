from multiprocessing import Process
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import glob

import environment

class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        r = np.random.rand()
        if r < 0.5:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1],
                                      friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1],
                                      friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return obj_pos, pixels

    def step(self, action_id):
        if action_id == 0:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 1:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 2:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 3:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})

def collect(idx, N, out_dir):
    env = Hw1Env(render_mode="offscreen")
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for i in range(N):
        env.reset()
        action_id = np.random.randint(4)
        _, img_before_sample = env.state()
        imgs_before[i] = img_before_sample
        env.step(action_id)
        obj_pos, img_after_sample = env.state()
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs_after[i] = img_after_sample
        print(f"Process {idx} collected {i+1}/{N} samples")
        
    torch.save(positions, os.path.join(out_dir, f"positions_{idx}.pt"))
    torch.save(actions, os.path.join(out_dir, f"actions_{idx}.pt"))
    torch.save(imgs_before, os.path.join(out_dir, f"imgs_before_{idx}.pt"))
    torch.save(imgs_after, os.path.join(out_dir, f"imgs_after_{idx}.pt"))

if __name__ == "__main__":
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    processes = []
    for i in range(20):
        p = Process(target=collect, args=(i, 50, temp_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    temp_positions_files = sorted(glob.glob(os.path.join(temp_dir, "positions_*.pt")))
    temp_actions_files = sorted(glob.glob(os.path.join(temp_dir, "actions_*.pt")))
    temp_imgs_before_files = sorted(glob.glob(os.path.join(temp_dir, "imgs_before_*.pt")))
    temp_imgs_after_files = sorted(glob.glob(os.path.join(temp_dir, "imgs_after_*.pt")))
    
    all_positions = [torch.load(f) for f in temp_positions_files]
    all_actions = [torch.load(f) for f in temp_actions_files]
    all_imgs_before = [torch.load(f) for f in temp_imgs_before_files]
    all_imgs_after = [torch.load(f) for f in temp_imgs_after_files]
    
    positions = torch.cat(all_positions, dim=0)
    actions = torch.cat(all_actions, dim=0)
    imgs_before = torch.cat(all_imgs_before, dim=0)
    imgs_after = torch.cat(all_imgs_after, dim=0)
    
    num_samples = positions.shape[0]
    indices = torch.randperm(num_samples)
    
    train_end = int(0.8 * num_samples)
    val_end = int(0.9 * num_samples)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_positions = positions[train_indices]
    train_actions = actions[train_indices]
    train_imgs_before = imgs_before[train_indices]
    train_imgs_after = imgs_after[train_indices]
    train_imgs = train_imgs_before.clone()
    
    val_positions = positions[val_indices]
    val_actions = actions[val_indices]
    val_imgs_before = imgs_before[val_indices]
    val_imgs_after = imgs_after[val_indices]
    val_imgs = val_imgs_before.clone()
    
    test_positions = positions[test_indices]
    test_actions = actions[test_indices]
    test_imgs_before = imgs_before[test_indices]
    test_imgs_after = imgs_after[test_indices]
    test_imgs = test_imgs_before.clone()
    
    os.makedirs("training_set", exist_ok=True)
    os.makedirs("validation_set", exist_ok=True)
    os.makedirs("test_set", exist_ok=True)
    
    torch.save(train_positions, os.path.join("training_set", "positions.pt"))
    torch.save(train_actions, os.path.join("training_set", "actions.pt"))
    torch.save(train_imgs, os.path.join("training_set", "imgs.pt"))
    torch.save(train_imgs_before, os.path.join("training_set", "imgs_before.pt"))
    torch.save(train_imgs_after, os.path.join("training_set", "imgs_after.pt"))
    
    torch.save(val_positions, os.path.join("validation_set", "positions.pt"))
    torch.save(val_actions, os.path.join("validation_set", "actions.pt"))
    torch.save(val_imgs, os.path.join("validation_set", "imgs.pt"))
    torch.save(val_imgs_before, os.path.join("validation_set", "imgs_before.pt"))
    torch.save(val_imgs_after, os.path.join("validation_set", "imgs_after.pt"))
    
    torch.save(test_positions, os.path.join("test_set", "positions.pt"))
    torch.save(test_actions, os.path.join("test_set", "actions.pt"))
    torch.save(test_imgs, os.path.join("test_set", "imgs.pt"))
    torch.save(test_imgs_before, os.path.join("test_set", "imgs_before.pt"))
    torch.save(test_imgs_after, os.path.join("test_set", "imgs_after.pt"))
    
    print("Data collection, merging, and 80/10/10 splitting complete.")
