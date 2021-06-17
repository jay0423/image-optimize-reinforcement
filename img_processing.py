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
import math

from ipywidgets import interact, interactive, fixed, RadioButtons
import ipywidgets as widgets
from IPython.display import display


class ImgProcessing:

    def __init__(self, img_address_list):
        self.img_address_list = img_address_list #初期画像のアドレス

        #アドレスから画像を取得
        img_original_list = []
        for i, img in enumerate(img_address_list):
            img_original_list.append(cv2.imread(img))
        self.img_original_list = img_original_list #初期画像リスト

        self.img_cleaned_list = [] #クリーンアップされた画像リスト
        self.img_processed_list = [] #画像処理された画像リスト

        self.img_synthetic = None #合成写真


    #複数画像のサイズを合わせる
    def cleaning(self, img_list):
        alpha = 5 #ずらす値
        cleaned_img_list = []
        for img in img_list:
            # cleaned_img_list.append(img[:113, 400:])
            cleaned_img_list.append(img[:100, 380:689])
        return cleaned_img_list


    # 赤だけ残した画像を出力
    def get_mask_bgr(self, bgr_img, get_color="r", m_min=200, m_max=255, frame=True, filtering=True):
        
        # 3色に分離（チャンネルを分ける）
        b, g, r = cv2.split(bgr_img)

        # get_colorから必要なbgrを取得する．
        color_list = ["b", "g", "r"]
        for i, c in enumerate(color_list):
            if c in sorted(get_color):
                color_list[i] = ""
        except_color_bgr_list = [b, g, r]
        except_color_list = [e for e, c in zip(except_color_bgr_list, color_list) if c != ""]

        # 指定されたカラーをマスクする処理
        except_img =bgr_img.copy()
        for color in except_color_list:
            _, mask_img = cv2.threshold(color, m_min, m_max, cv2.THRESH_BINARY_INV) # マスク（黒にする）
            except_img = cv2.bitwise_and(except_img, except_img, mask=mask_img) # 除外

        #　初期画像の白部分の取得と合成
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
    def processing(self, img_list, get_color="r", m_min=200, m_max=255, frame=True, filtering=True):
        img_processed_list = [] #画像処理が完了した画像リスト
        for img in img_list:
            img_processed_list.append(self.get_mask_bgr(img, get_color, m_min, m_max, frame, filtering)) #赤色を摘出し格納
        return img_processed_list

    
    def img_synthetic_how_j(self, img_list):
        # 新たな画像処理アルゴリズム
        img_synthetic= img_list[0].copy()
        for i in range(len(img_list)-1):
            img_i = img_list[i+1]
            for j in range(len(img_synthetic)):
                for k in range(len(img_synthetic[i])):
                    bgr_synsthetic = img_synthetic[j][k]
                    bgr_i = img_i[j][k]
                    zeros_synthetic = np.count_nonzero(bgr_synsthetic < 10) # 10より小さい値の数を取得0~3
                    zeros_i = np.count_nonzero(bgr_i < 10)
                    # ０の数に応じてrgb決定する
                    if zeros_synthetic == 0:
                        if zeros_i == 0:
                            get_bgr = np.maximum(bgr_synsthetic, bgr_i)
                        elif zeros_i == 1:
                            get_bgr = bgr_i
                        elif zeros_i == 2:
                            get_bgr = bgr_i
                        elif zeros_i == 3:
                            get_bgr = bgr_synsthetic
                    elif zeros_synthetic == 1:
                        if zeros_i == 0:
                            get_bgr = bgr_synsthetic
                        elif zeros_i == 1:
                            get_bgr = np.minimum(bgr_synsthetic, bgr_i)
                        elif zeros_i == 2:
                            get_bgr = bgr_i
                        elif zeros_i == 3:
                            get_bgr = bgr_synsthetic
                    elif zeros_synthetic == 2:
                        if zeros_i == 0:
                            get_bgr = bgr_synsthetic
                        elif zeros_i == 1:
                            get_bgr = bgr_synsthetic
                        elif zeros_i == 2:
                            get_bgr = np.maximum(bgr_synsthetic, bgr_i)
                        elif zeros_i == 3:
                            get_bgr = bgr_synsthetic
                    elif zeros_synthetic == 3:
                        if zeros_i == 0:
                            get_bgr = bgr_i
                        elif zeros_i == 1:
                            get_bgr = bgr_i
                        elif zeros_i == 2:
                            get_bgr = bgr_i
                        elif zeros_i == 3:
                            get_bgr = [0, 0, 0]

                    img_synthetic[j][k] = get_bgr # 変更
        return img_synthetic

    

    #複数画像を合成する．
    def synthetic(self, img_list, gry=True, inv=False, how="sum"):
        img_list = img_list.copy() # メモリを分けるための処理
        # グレースケール化
        if gry:
            for i, img in enumerate(img_list):
                img_list[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 編集処理の選択
        if how == "sum":
            img_synthetic = sum(img_list)
        elif how == "average":
            img_synthetic = img_list[0]//3 + img_list[1]//3 + img_list[2]//3
        elif how == "max":
            img_synthetic = img_list[0]
            # print(img_list[1][50])
            for i in range(len(img_list)-1):
                img_synthetic = np.maximum(img_synthetic,img_list[i+1])
        elif how == "min":
            img_synthetic = img_list[0]
            for i in range(len(img_list)-1):
                img_synthetic = np.minimum(img_synthetic,img_list[i+1])
        elif how == "j":
            img_synthetic = self.img_synthetic_how_j(img_list)

        if inv:
            img_synthetic = cv2.bitwise_not(img_synthetic)
        return img_synthetic
        
    

    def main(self):
        self.img_cleaned_list = self.cleaning(self.img_original_list)
        self.img_processed_list = self.processing(self.img_cleaned_list, get_color="r", m_min=200, m_max=255, frame=True, filtering=False)
        self.img_synthetic = self.synthetic(self.img_processed_list, gry=False, inv=False, how="j")
        print("画像を合成させました．")


    #複数画像を同時に表示する．
    def show_img(self, target=""):
        
        #対象の画像リストを取得
        if target == "before":
            img_list = self.img_original_list
        elif target == "cleaned":
            img_list = self.img_cleaned_list
        elif target == "after":
            img_list = self.img_processed_list
        elif target == "all":
            img_list = self.img_original_list + self.img_cleaned_list + self.img_processed_list
        else:
            print("error: 対象画像群を指定してください．\ntarget = 'before', 'cleaned', 'after' or 'all'")
            return
        
        title_list = self.img_address_list # 画像のタイトル名を取得

        N = len(img_list)
        col = 2
        row = math.ceil(N / col)

        fig = plt.figure(target)
        for i, img, title in zip(range(N), img_list, title_list):
            fig.add_subplot(row, col, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off') #座標軸非表示
            plt.title(title) #タイトル設定
        plt.show()
    

    # 最終結果の表示
    def result(self):
        # 最終合成写真を表示する
        fig = plt.figure("Synthetic Image")
        plt.imshow(cv2.cvtColor(self.img_synthetic, cv2.COLOR_BGR2RGB))
        plt.axis('off') #座標軸非表示
        plt.show()


if __name__ == "__main__":

    a = ImgProcessing(["sample_img/x0.png", "sample_img/y0.png", "sample_img/hakai0.png", "sample_img/y90.png", "sample_img/hakai90.png"])
    # a = ImgProcessing(["sample_img/x0.png", "sample_img/y0.png"])
    a.main()
    # a.show_img("before")
    # a.show_img("cleaned")
    a.show_img("after")
    # a.show_img("all")
    a.result()