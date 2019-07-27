import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import numpy as np

""" 摄像头打开失败的mBox"""
from tkinter import messagebox as mBox
import predict
import cv2
from PIL import Image, ImageTk
import threading
import time


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("绿牌", "#55FF55"), "yello": ("黄牌", "#FFFF00"), "blue": ("蓝牌", "#6666FF")}

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)

        win.title("车牌识别")
        win.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌位置：').grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)
        from_vedio_ctl = ttk.Button(frame_right2, text="来自摄像头", width=20, command=self.from_vedio)
        close_vedio_ctl = ttk.Button(frame_right2, text="关闭摄像头", width=20, command=self.close_vedio)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_vedio_ctl.pack(anchor="se", pady="5")
        from_pic_ctl.pack(anchor="se", pady="5")
        close_vedio_ctl.pack(anchor="se", pady="5")
        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def show_roi(self, r, roi, color):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except:
                self.color_ctl.configure(state='disabled')
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')

    def from_vedio(self):
        """        if self.thread_run:
            mBox.showwarning('提示', '摄像头已打开！')
            return
        """
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)  # 打开笔记本内置摄像头
            if not self.camera.isOpened():  # 是否打开成功
                mBox.showwarning('警告', '摄像头打开失败！')
                self.camera = None
                return
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))  # 构造线程运行vedio_thread方法
        self.thread.setDaemon(True)  # 设置为后台线程
        self.thread.start()
        # self.thread_run = True

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            # read a frame of camere
            ret, img_bgr = self.camera.read()
            # 如果读取成功,ret==true
            if not ret:
                mBox.showwarning('警告', '获取图像失败！')
                break
            self.gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # call classifier for face detection
            self.faces = faceCascade.detectMultiScale(
                        self.gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
            )
            # draw a rectangle
            for (x, y, w, h) in self.faces:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 1:   # 0.5s 识别一次
#                try:

                recv_r, recv_roi, recv_color = self.predictor.predict(img_bgr)

#                except:
#                    continue
                self.show_roi(recv_r, recv_roi, recv_color)
                predict_time = time.time()
        if(self.thread_run != False):
            self.camera.release()
            # avoid camera.release() twice
        # self.camera.release()
        cv2.destroyAllWindows()
        print("run end")

    def close_vedio(self):
        self.thread_run = False
        try:
            self.camera.release()
        except:
            self.thread_run = False
        cv2.destroyAllWindows()
        self.camera = None

    def from_pic(self):
        r: list
        roi: np.ndarray
        color: str
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if self.pic_path:
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)

            r, roi, color = self.predictor.predict(img_bgr)     # r: list   roi:ndarray    color: str

            self.show_roi(r, roi, color)


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()

    # the path of module of face_detection in opencv
    # cascPath = "D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
    # copy to the current dir
    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    surface = Surface(win)

    # 加载训练好的模型
    try:
        predict.model = predict.load_carplate_CNN()
    except:
        None
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()
