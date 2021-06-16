# coding: UTF-8
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import skimage
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_local
from skimage.exposure import histogram, adjust_gamma
from skimage.morphology import square
from skimage import measure, color, morphology

import cv2
import numpy as np

from ipywidgets import interact, interactive, fixed, RadioButtons
import ipywidgets as widgets
from IPython.display import display


class ImgProcessing:

    def __init__(self, img_address_list):
        self.img_address_list = img_address_list

        img_list = []
        for i, img in enumerate(img_address_list):
            img_list.append(cv2.imread(img))
        self.img_list = img_list
        self.img_cleaned_list = []
        self.img_processed_list = []


    # def show_img(self, img, vmin=0, vmax=255, title=None):
    #     if img.ndim == 3:
    #         #カラー(ndim=次元数=3 -> RGBの３色)
    #         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #CV2のRGB３色に変換
    #         plt.imshow(rgb_img, vmin=vmin, vmax=vmax) #変換後画像をplotへ設定
    #     elif img.ndim == 2:
    #         #グレースケール（ndim=次元数=2 -> 白黒の２色）
    #         plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax) #変換後画像をplotへ設定
    #     plt.axis('off') #座標軸非表示
    #     plt.title(title) #タイトル設定
    #     # plt.show() #グラフ表示

        
    #画像表示（比較）
    def show_img_compare(self, src, dst, vmin=0, vmax=255):
        # 元画像表示
        plt.subplot(121)
        self.show_img(src, vmin, vmax, title='Before')
            
        # 返還後画像表示
        plt.subplot(122)
        self.show_img(dst, vmin, vmax, title='After')



    #複数画像を同時に表示する．
    def show_img(self, target=""):
        
        #対象の画像リストを取得
        if target == "before":
            img_list = self.img_list
        elif target == "cleaned":
            img_list = self.img_cleaned_list
        elif target == "after":
            img_list = self.img_processed_list
        elif target == "all":
            img_list = self.img_list + self.img_cleaned_list + self.img_processed_list
        else:
            print("error: 対象画像群を指定してください．\ntarget = 'before', 'cleaned', 'after' or 'all'")
            return

        
        # 画像のタイトル名を取得
        title_list = self.img_address_list

        N = len(img_list)
        col = 2
        row = round(N / col)

        fig = plt.figure(target)
        for i, img, title in zip(range(N), img_list, title_list):
            fig.add_subplot(row, col, i+1)
            plt.imshow(img)
            plt.axis('off') #座標軸非表示
            plt.title(title) #タイトル設定
        plt.show()

 
    #赤だけ残した画像を出力
    def get_red(self, bgr_img, m_min=200, m_max=255, frame=True, green=True, filtering=True):
        #3色に分離（チャンネルを分ける）
        b, g, r = cv2.split(bgr_img)
        
        _, mask_b_img = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY_INV) #blueをマスク（黒にする）
        _, mask_g_img = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY_INV) #greenをマスク（黒にする）
        except_b_img = cv2.bitwise_and(bgr_img, bgr_img, mask=mask_b_img) #元画像と合成して青を除外（白の座標のみ元画像を復元）
        except_g_img = cv2.bitwise_and(except_b_img, except_b_img, mask=mask_g_img) #さらに緑を除外

        #redをマスク（白にする）
        _, mask_r_img = cv2.threshold(r, m_min, m_max, cv2.THRESH_BINARY)

        #元画像と合成してredのみにする（白の座標のみ元画像を復元）
        only_r_img = cv2.bitwise_and(except_b_img, except_b_img, mask= mask_r_img)

        #元画像と合成してredのみにする（2）
        only_r_img2 = cv2.bitwise_and(except_g_img, except_g_img, mask= mask_r_img)
        
        #　外枠の取得と合成
        if frame:
            # グレースケール化
            gry = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            # 閾値135で二値化（白でマスク）
            ret, thresh = cv2.threshold(gry, 230, 255, 0)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            only_r_img = thresh + only_r_img
            only_r_img2 = thresh + only_r_img2
        
        #境界線が粗いのでフィルタをかける(境界を曖昧にする)
        if filtering:
            only_r_img = cv2.medianBlur(only_r_img, 5)
            only_r_img2 = cv2.medianBlur(only_r_img2, 5)
            
        #出力
        if green:
            return only_r_img2
        else:
            return only_r_img


    #複数画像のサイズを合わせる
    def cleaning(self, img_list):
        alpha = 5 #ずらす値
        cleaned_img_list = []
        for img in img_list:
            # cleaned_img_list.append(img[:113, 400:])
            cleaned_img_list.append(img[:100, 380:])
        return cleaned_img_list


    #各画像の赤い部分を抽出し合成する．
    def processing(self, img_list):
        img_processed_list = [] #画像処理が完了した画像リスト
        for img in img_list:
            img_processed_list.append(self.get_red(img)) #赤色を摘出し格納
        return img_processed_list
    

    #複数画像を合成する．
    def synthetic(self):
        pass
    

    def main(self):
        self.img_cleaned_list = self.cleaning(self.img_list)
        self.img_processed_list = self.processing(self.img_cleaned_list)
        print("画像を合成させました．")


if __name__ == "__main__":

    a = ImgProcessing(["sample_img/x0.png", "sample_img/y0.png", "sample_img/hakai0.png"])
    a.main()
    # a.show_img()
    # a.show_img("before")
    # a.show_img("cleaned")
    # a.show_img("after")
    a.show_img("all")


    # #ずらす値
    # alpha = 5
    # im_x0 = cv2.imread('sample_img/x0.png')
    # im_y0 = cv2.imread('sample_img/y0.png')
    # im_hakai = cv2.imread('sample_img/hakai0.png')
    # im_x0 = im_x0[:113, 400:]
    # im_y0 = im_y0[:113, 400-alpha:689-alpha]
    # im_hakai = im_hakai[:113, 400-alpha:689-alpha]
    # im = (im_x0 + im_y0 + im_hakai) - 100
    #im_x0_red = a.get_red(im_x0, frame=True, filtering=False)
    # show_img_compare(im_x0, im_x0_red)
    #a.show_img(im_x0_red)