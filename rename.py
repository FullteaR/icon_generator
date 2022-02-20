import cv2
import os
import numpy as np
from tqdm.auto import tqdm
import ray


# In[2]:


ray.init()


# In[3]:


def grayCheck(img_path):
    img=cv2.imread(img_path)
    histr=[cv2.calcHist([img],[j],None,[256],[0,256]) for j in range(3)]
    s=0
    for j in range(3):
        d=(histr[(j+1)%3]-histr[j])**2
        d=d/(np.sum(d)+1)
        s+=d
    s=s/3
    return np.sum(s)<0.1

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

@ray.remote
def grayDelete(img_path):
    try:
        if grayCheck(img_path):
            print("delete", img_path)
            os.remove(img_path)
    except:
        print("Error", img_path)
        print("delete", img_path)
        os.remove(img_path)


# In[4]:


img_paths=[os.path.join("input", path) for path in os.listdir("./input")]
ray_ids=[grayDelete.remote(img_path) for img_path in img_paths if os.path.isfile(img_path)]
grays=[]
for x in tqdm(to_iterator(ray_ids), total=len(ray_ids)):
    pass
