import numpy as np
import cv2

# NumPy 배열 로드
image_data = np.load(r'C:\Users\user\Desktop\record_video\save\20250924_163939\crop\20250924_164103\depth_raw_frames\frame_000010.npy')

# 정규화 (0~255 범위로 변환)
image_norm = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
image_norm = image_norm.astype(np.uint8)

# 컬러맵 적용 (JET, HOT, TURBO 등 가능)
image_color = cv2.applyColorMap(image_norm, cv2.COLORMAP_JET)

# OpenCV 창에 표시
cv2.imshow("Depth Frame (Colormap)", image_color)
cv2.waitKey(0)   # 키 입력 대기
cv2.destroyAllWindows()
