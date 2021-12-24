#โปรแกรมนี้เป็นโปรแกรมตรวจจับใบหน้าจากกล้อง webcam
#หรือ video บนคอมพิวเตอร์ อาจมีการปรับแต่งให้ตรวจจับอย่างอื่นเพิ่มเติมได้ในภายหลัง
#สมาชิกกลุ่มใช้ visual studio code ในการสร้างและทดสอบโปรแกรม

#ติดตั้งเครื่องมือ 
    #1. ติดตั้ง python
    #2. ติดตั้ง opencv โดยใช้คำสั่ง "pip install opencv-python" ใน cmd หรือ terminal
    #3. เมื่อดาวน์โหลดโค้ดแล้วต้องทำให้ไฟล์ "Face_Detector.py" และ haarcascade_frontalface_default.xml 
        #อยู่ในโฟลเดอร์เดียวกัน
    
#วิธีการใช้งาน 
    #รันโปรแกรม โดยใช้คำสั่ง "python Face_Detector.py" ใน cmd หรือ terminal
    #หากต้องการออกจากโปรแกรมให้กดปุ่ม Q หรือ q

import cv2

# Load some pre-trained data on face frontals from opencv (haar cassade algorithm)
# เทรนด้วยข้อมูลจากไฟล์ xml (เฉพาะใบหน้าด้านหน้า)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
                    #call opencv
                        #call this function to make classifier

# choose image to detect faces from webcam 
# you can detect from video by change (0) to your video name like "(cute_cat.mp4)"
webcam = cv2.VideoCapture(0)


while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    # ทำให้ภาพจาก frame เป็นสีเทาเพื่อให้ algorithm ง่ายขึ้น และ ลดเวลาการประมวนผลลง
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces 'face_coordinates' now is mean the position that use to tell area of face like 336 115... (UpLeft width height)->(x y w h)     
    # หากพบสิ่งที่มีลักษณะเหมือนในข้อมูลนำไปเทรน จะเก็บตำแหน่งของ area ของสิ่งนั้นไว้ โดยเก็บเป็นเลข 4 จำนวน (area เป็นรูปสี่เหลี่ยม)
    # 2 จำนวนแรกคือตำแหน่งแกน x, y ของมุมซ้ายบนของสี่เหลี่ยม อีก 2 จำนวนคือความกว้างและความสูงของสี่เหลี่ยม w, h                                                                                                                     
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    #Display images
    cv2.imshow('look at that chinese guy make some face detector',frame)

    # wait for press some key and continue if put the number in there that mean delay
    key = cv2.waitKey(1)

    #stop if we pressed "Q" or "q" | 81 and 113 is ASCII
    if key==81 or key==113:
        break

#Released the video capture object
webcam.release()

