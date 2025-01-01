from datasets.artpedia import download_artpedia_images,download_artpedia_zip
import sys

if __name__=="__main__":
    output_path=sys.argv[1]
    download_artpedia_zip(output_path)
    download_artpedia_images(output_path,write_stat_dict=True)