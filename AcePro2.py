'''
Author: SingleBiu
Date: 2024-10-17
Description: YuNet DNN face detect with Insta360 Ace Pro 2
'''
import os
import cv2 as cv
import numpy as np

# ========== 1. 加载 YuNet 模型 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'face_detection_yunet_2023mar.onnx')

print(f"[INFO] 模型路径: {MODEL_PATH}")
print(f"[INFO] 文件存在: {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"找不到 YuNet 模型: {MODEL_PATH}\n"
        "请从 OpenCV Zoo 下载 face_detection_yunet_2023mar.onnx"
    )

# 初始化 YuNet 检测器
# 参数：模型路径、配置文件（留空）、初始输入尺寸、置信度阈值、NMS阈值、top_k
face_detector = cv.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),        # 初始输入尺寸，后面每帧会更新
    score_threshold=0.6,  # 置信度阈值：越高越严格
    nms_threshold=0.3,    # NMS 阈值：抑制重叠框
    top_k=5000
)
print("[OK] YuNet 模型加载成功")


# ========== 2. YuNet 人脸检测函数 ==========
def face_detect_yunet(img):
    h, w = img.shape[:2]

    # 关键：每帧更新检测器的输入尺寸
    face_detector.setInputSize((w, h))

    # 执行检测
    # faces 形状: [num_faces, 15]
    # 每行: [x, y, w, h, 右眼x, 右眼y, 左眼x, 左眼y, 鼻尖x, 鼻尖y, 右嘴角x, 右嘴角y, 左嘴角x, 左嘴角y, 置信度]
    retval, faces = face_detector.detect(img)

    if faces is not None:
        for face in faces:
            # 提取边界框
            x, y, fw, fh = face[:4].astype(int)

            # 越界裁剪
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            # 画框
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 显示置信度
            confidence = face[14]
            cv.putText(img, f"{confidence:.2f}", (x1, y1 - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow('result', img)


# ========== 3. 打开摄像头 ==========
def open_camera():
    for index in [0, 1, 2]:
        cap = cv.VideoCapture(index, cv.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[OK] 成功打开摄像头，设备索引: {index}")
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv.CAP_PROP_FPS, 30)
                return cap
            cap.release()
        else:
            cap.release()
        print(f"[..] 设备索引 {index} 不可用，尝试下一个...")
    return None


# ========== 4. 主流程 ==========
def main():
    cap = open_camera()
    if cap is None:
        print("[ERR] 未能打开摄像头，请检查 Ace Pro 2 是否已切换到 Webcam 模式。")
        return

    while True:
        flag, frame = cap.read()
        if not flag:
            print("[ERR] 读取画面失败，退出。")
            break

        face_detect_yunet(frame)

        if cv.waitKey(1) & 0xFF == ord('m'):
            break

    cap.release()
    cv.destroyAllWindows()
    print("[OK] 已释放摄像头，程序退出。")


if __name__ == '__main__':
    main()