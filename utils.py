import numpy as np
import cv2

'''
定義剪裁image的function
x0 = img 左上角橫座標
y0 = img 左上角縱座標
wc = 剪裁寬度
hc = 剪裁高度
'''
crop_img = lambda img, x0, y0, wc, hc: img[x0:x0 + wc, y0:y0 + hc]

def random_crop(img, a_ratio, hw_noise):
    '''
    隨機剪裁
    img為輸入圖片
    a_ratio為剪裁面積與原圖面積比
    hw_noise為隨機產生之剪裁範圍誤差占原圖高寬比，增加泛化能力
    '''
    h, w = img.shape[:2] #numpy array row:img_h column:img_w
    hw_delta = np.random.uniform(-hw_noise, hw_noise)
    hw_mult = 1 + hw_delta
    
    #剪裁寬度，取整數
    w_crop = int(round(w * np.sqrt(a_ratio * hw_mult)))
    if w_crop > w:
        w_crop = w #剪裁寬度不可超過原圖寬度
    
    h_crop = int(round(h * np.sqrt(a_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h
    
    #於圖片左上角隨機生成剪裁座標
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_img(img, x0, y0, w_crop, h_crop)