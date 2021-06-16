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
        self.img_address_list = img_address_list #初期画像のアドレス

        #アドレスから画像を取得
        img_list = []
        for i, img in enumerate(img_address_list):
            img_list.append(cv2.imread(img))
        self.img_list = img_list #初期画像リスト

        self.img_cleaned_list = [] #クリーンアップされた画像リスト
        self.img_processed_list = [] #画像処理された画像リスト


    #複数画像のサイズを合わせる
    def cleaning(self, img_list):
        alpha = 5 #ずらす値
        cleaned_img_list = []
        for img in img_list:
            # cleaned_img_list.append(img[:113, 400:])
            cleaned_img_list.append(img[:100, 380:])
        return cleaned_img_list


    # 赤だけ残した画像を出力
    def get_mask_bgr(self, bgr_img, get_color="g", m_min=200, m_max=255, frame=True, filtering=True):
        
        # 3色に分離（チャンネルを分ける）
        b, g, r = cv2.split(bgr_img)

        # get_colorから必要なbgrを取得する．
        color_list = ["b", "g", "r"]
        for i, c in enumerate(color_list):
            if c in sorted(get_color):
                color_list[i] = ""
        except_color_bgr_list = [b, g, r]
        except_color_list = [e for e, c in zip(except_color_bgr_list, color_list) if c != ""]
        # for e, c in zip(except_color_bgr_list, color_list):
        #     if c != "":
        #         except_color_list.append(e)

        # 指定されたカラーをマスクする処理
        except_img =bgr_img.copy()
        for color in except_color_list:
            _, mask_img = cv2.threshold(color, 200, 255, cv2.THRESH_BINARY_INV) # マスク（黒にする）
            except_img = cv2.bitwise_and(except_img, except_img, mask=mask_img) # 除外

        #　外枠の取得と合成
        if frame:
            # グレースケール化
            gry = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            # 閾値135で二値化（白でマスク）
            ret, thresh = cv2.threshold(gry, 230, 255, 0)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            except_img = thresh + except_img
        
        #境界線が粗いのでフィルタをかける(境界を曖昧にする)
        if filtering:
            except_img = cv2.medianBlur(except_img, 5)
            
        return except_img


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
        
        title_list = self.img_address_list # 画像のタイトル名を取得

        N = len(img_list)
        col = 2
        row = round(N / col)

        fig = plt.figure(target)
        for i, img, title in zip(range(N), img_list, title_list):
            fig.add_subplot(row, col, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off') #座標軸非表示
            plt.title(title) #タイトル設定
        plt.show()


if __name__ == "__main__":

    a = ImgProcessing(["sample_img/x0.png", "sample_img/y0.png", "sample_img/hakai0.png"])
    a.main()
    # a.show_img()
    a.show_img("before")
    a.show_img("cleaned")
    a.show_img("after")
    # a.show_img("all")


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