"""
some function to crop, rotate, change HSV from image.
"""
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
    hw_noise為隨機產生之剪裁範圍誤差占原圖高寬比的範圍，增加泛化能力
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
        h_crop = h#剪裁高度不可超過原圖高度
    
    #於圖片左上角隨機生成剪裁座標
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_img(img, x0, y0, w_crop, h_crop)

def rotate_img(img, angle, edge):
    '''
    照片旋轉
    img為輸入圖片
    angle為逆時針旋轉角度
    edge為bool表示是否裁去黑邊
    '''
    h, w = img.shape[:2]
    angle %= 360
    #定義旋轉矩陣
    rot_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

    rot_img = cv2.warpAffine(img, rot_matrix, (w, h))

    if edge:
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
                 
        theta_ = angle_crop * np.pi / 180

        hw_ratio = float(h) / float(w)

        tan_theta = np.tan(theta_)
        numerator = np.cos(theta_) + np.sin(theta_) * tan_theta
        
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1

        crop_mult = numerator / denominator

        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        rot_img = crop_img(rot_img, x0, y0, w_crop, h_crop)

    return rot_img

def random_rotate_img(img, rand_angle, p_crop):
    '''
    隨機旋轉照片
    img為原照片
    rand_angle為隨機旋轉範圍（在+-範圍間隨機取值）
    p_crop為要進行裁去黑邊的比例
    '''
    angle = np.random.uniform(-rand_angle, rand_angle)
    crop = False if np.random.random() > p_crop else True
    return rotate_img(img, angle, crop)

def hsv_transform(img, hue_ratio, sat_ratio, val_ratio):
    '''
    定義hsv變換
    img為原照片
    hue_ratio為色調變化比例
    sat_ratio為飽和度變化比例
    val_ratio為明亮度變化比例
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_ratio) % 180
    img_hsv[:, :, 1] *= sat_ratio
    img_hsv[:, :, 2] *= val_ratio
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2RGB)

def random_hsv_transform(img, hue_rand, sat_rand, val_rand):
    '''
    隨機變換hsv
    hue_rand為隨機取值色調變換比例範圍
    sat_rand為隨機取值飽和度比例範圍
    val_rand為隨機取值明亮度比例範圍
    '''
    hue_delta = np.random.randint(-hue_rand, hue_rand)
    sat_mult = 1 + np.random.uniform(-sat_rand, sat_rand)
    val_mult = 1 + np.random.uniform(-val_rand, val_rand)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

def gamma_transform(img, gamma):
    '''
    定義gamma變換
    '''
    gamma_table = [np.power(x / 255., gamma) * 255. for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_rand):
    '''
    隨機gamma變換
    '''
    log_gamma_rand = np.log(gamma_rand)
    alpha = np.random.uniform(-log_gamma_rand, log_gamma_rand)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
