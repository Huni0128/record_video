import numpy as np
import matplotlib.pyplot as plt

# 이미지 데이터를 NumPy 배열로 읽어옵니다.
image_data = np.load(r'C:\Users\user\Desktop\record_video\save\farm_record\20250924_102620\depth_image_raw_20250912_023416.npy')


# 이미지를 시각화합니다.
plt.imshow(image_data, cmap='gray')  # cmap을 원하는 색상 맵으로 변경할 수 있습니다.
plt.title('Image')
plt.axis('off')  # 축을 비활성화하여 이미지에 불필요한 눈금을 표시하지 않습니다.
plt.show()