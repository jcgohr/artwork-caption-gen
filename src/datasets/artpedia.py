from urllib.parse import unquote
from tqdm import tqdm
from PIL import Image 
import urllib.request
import zipfile 
import shutil
import math
import json
import os
import re



def normalize_image(img_path,max_pixels=1000):
    """
    Resizes to fit within a 1000x1000 frame
    """
    Image.MAX_IMAGE_PIXELS = 2000000000 
    im = Image.open(img_path)
    w,h = im.size
    # If no resizing is required, just return
    if w<=max_pixels and h<=max_pixels:
        return
     
    max_pix_dim=w
    if h>w:
        max_pix_dim=h
    
    scaling_factor=math.ceil(max_pix_dim/max_pixels)
    im=im.resize((w//scaling_factor,h//scaling_factor))
    im.save(img_path)
    im.close()
    


def clean_filename(filename):
    """Strips all characters that cannot be in a filename."""
    return re.sub(r'[\/:*?"<>|]', '', filename)

ARTPEDIA_LINK="https://aimagelab.ing.unimore.it/imagelab/uploadedFiles/artpedia.zip"
ZIP="artpedia.zip"
IMG_DIR="images/"

def download_artpedia_zip(output_dir:str,redownload=False):

    if os.path.exists(output_dir) and not redownload:
        print("The zip file has already been downloaded")
        return
    
    # Make the dataset directory
    os.makedirs(output_dir,exist_ok=True)
    
    # Join the zip file with the output_dir
    zip_path=os.path.join(output_dir,ZIP)
    
    # Extract the zip
    with urllib.request.urlopen(ARTPEDIA_LINK) as response, open(zip_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
            
    # Remove the zip
    os.remove(zip_path)
    
def download_artpedia_images(output_dir:str,write_stat_dict=False):
    with open(os.path.join(output_dir,"artpedia.json"),encoding="utf-8") as artpedia_file:
        artpedia_dict=json.load(artpedia_file)
    
    stat_dict={}
    
    # Add user agent 
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0')]
    urllib.request.install_opener(opener)
    
    img_directory=os.path.join(output_dir,IMG_DIR)
    
    os.makedirs(img_directory,exist_ok=True)
    
    for key in tqdm(list(artpedia_dict.keys()),desc="Downloading images from wikimedia"):        
        url=artpedia_dict[key]["img_url"]
        save=os.path.join(img_directory,clean_filename(os.path.basename(unquote(url))))
        if os.path.exists(save):
            continue
        artpedia_dict[key]["file_path"]=save
        try:
            urllib.request.urlretrieve(url,save)
        except urllib.request.HTTPError as e:
            # If the resource does not exist remove the entry in artpedia.json
            if e.code==404:
                del artpedia_dict[key]
                
            if e.code in stat_dict:
                stat_dict[e.code].append(url)
            else:
                stat_dict[e.code]=[url]
                
    # Write the stat dict if write_stat_dict
    if write_stat_dict:
        with open(os.path.join(output_dir,"stats.json"),"w",encoding="utf-8") as stat_dict_file:
            stat_dict_file.write(json.dumps(stat_dict,indent=4))
            
    # Write the update artpedia dictionary back to artpedia.json (we added the file paths and removed 404'd images)
    with open(os.path.join(output_dir,"artpedia.json"),"w",encoding="utf-8") as artpedia_file:
            artpedia_file.write(json.dumps(artpedia_dict,indent=4,ensure_ascii=False))

def download_artpedia(output_dir:str,):
    download_artpedia_zip(output_dir)
    download_artpedia_images(output_dir)