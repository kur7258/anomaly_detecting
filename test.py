import cv2
import os

# 입력 동영상 파일 경로
video_path = 'test_vid.MOV'
# 출력 폴더
output_folder = 'frames'
os.makedirs(output_folder, exist_ok=True)

# 비디오 열기
video = cv2.VideoCapture(video_path)
success, frame = video.read()
count = 0

while success:
    frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
    cv2.imwrite(frame_path, frame)
    success, frame = video.read()
    count += 1

print(f"총 {count}개의 프레임이 저장되었습니다.")
video.release()
cv2.destroyAllWindows()
