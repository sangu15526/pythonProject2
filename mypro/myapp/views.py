from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from django.contrib.auth.models import User
# Machine Learning Packages
import numpy as np
import cv2
import os
from PIL import Image
import datetime


# Machine Learning Packages


# Create your views here.

@login_required(login_url='login-h')
def home(request):
    return render(request, 'index.html')


def register(request):
    form = CreateUserForm()

    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            messages.success(request, 'Account was created for ' + username)

            return redirect('add_face')
    context = {'form': form}
    return render(request, 'register.html', context)


def loginpage(request):
    last_login.objects.all().delete()
    if request.method == 'POST':
        try:
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, username=username, password=password)
        except:
            messages.warning("Fill the details")

        if user is not None:
            login(request, user)
            var = last_login(username=username, password=password)
            var.save()
            if user.is_staff:
                return redirect('staff-home')
            return redirect('home')

        else:
            messages.info(request, 'Username OR password is incorrect')

    context = {}
    return render(request, 'login.html', context)


def add_face(request):
    if not os.path.exists('media/images'):
        os.makedirs('media/images')

    faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    count = 0

    face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    var = User.objects.last()
    face_id = var.id
    face_id = int(face_id)
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the images directory
            cv2.imwrite("media/images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        # Press Escape to end the program.
        k = cv2.waitKey(100) & 0xff
        if k < 30:
            break
        # Take 30 face samples and stop video. You may increase or decrease the number of
        # images. The more the better while training the model.
        elif count >= 30:
            break

    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()

    return render(request, 'photo.html')


def getImagesAndLabels():
    path = 'media/images/'
    detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml");
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        # convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids


def create_model(request):
    path = 'media/images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n[INFO] Training faces...")
    faces, ids = getImagesAndLabels()
    recognizer.train(faces, np.array(ids))
    # Save the model into the current directory.
    recognizer.write('data/trainer.yml')
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    return redirect('register')


def attend(request):
    var = User.objects.all()
    names = []
    names.append(None)
    for i in var:
        names.append(i.username)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read('data/trainer.yml')
    except:
        return redirect('register')

    face_cascade_Path = "data/haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(face_cascade_Path)

    new_model = tf.keras.models.load_model('data/assignmentmodel1.h5')

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0
    # names related to ids: The names associated to the ids: 1 for Mohamed, 2 for Jack, etc...
    # names = ['specs', 'reshma', 'res', 'None']  # add a name into this list
    # Video Capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    # Min Height and Width for the  window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    count = 0
    att = 0
    list1 = []
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id1, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            print(id1)
            out = np.expand_dims(img, axis=0)
            final_img = out / 255.0

            pred = new_model.predict(final_img)
            max_index = np.argmax(pred[0])
            print(max_index)
            values = ('bored', 'confusion', 'engaged', 'frustrated')

            predicted = values[max_index]
            print(pred[0])
            print(predicted)
            list1.append(predicted)
            if confidence < 100:
                print('id:', id1)
                print(names)
                id = names[id1]
                confidence = "  {0}%".format(round(100 - confidence))
                if att == 0:
                    person = User.objects.get(username=id)
                    att = 0
                    new = attendance.objects.filter(user=person).filter(date=datetime.datetime.today())

                    if len(new) == 0:
                        details = attendance(user=person, date=datetime.datetime.today(),
                                             Time=datetime.datetime.now().strftime("%H:%M:%S"))
                        details.save()
                        detail = attention(user=person, date=datetime.datetime.today(), emotion=predicted,
                                           Time=datetime.datetime.now().strftime("%H:%M:%S"))
                        detail.save()
                    else:
                        print('time', (datetime.datetime.now() - datetime.timedelta(minutes=60)).strftime("%H:%M:%S"))
                        new1 = \
                            attendance.objects.filter(user=person).filter(date=datetime.datetime.today()).order_by(
                                '-id')[0]
                        print(new1)
                        if new1.Time.strftime("%H:%M:%S") <= (
                                datetime.datetime.now() - datetime.timedelta(minutes=60)).strftime("%H:%M:%S"):
                            details = attendance(user=person, date=datetime.datetime.today(),
                                                 Time=datetime.datetime.now().strftime("%H:%M:%S"))
                            details.save()
                            detail = attention(user=person, date=datetime.datetime.today(), emotion=predicted,
                                               Time=datetime.datetime.now().strftime("%H:%M:%S"))
                            detail.save()
                        elif new1.Time.strftime("%H:%M:%S") <= (
                                datetime.datetime.now() - datetime.timedelta(minutes=15)).strftime("%H:%M:%S"):
                            details = attention(user=person, date=datetime.datetime.today(), emotion=predicted,
                                                Time=datetime.datetime.now().strftime("%H:%M:%S"))
                            details.save()
                        else:
                            print('false')
                    new2 = attention.objects.filter(user=person).filter(date=datetime.datetime.today()).order_by('-id')[
                        0]
                    if new2.Time.strftime("%H:%M:%S") <= (
                            datetime.datetime.now() - datetime.timedelta(minutes=15)).strftime("%H:%M:%S"):
                        details = attention(user=person, date=datetime.datetime.today(), emotion=predicted,
                                            Time=datetime.datetime.now().strftime("%H:%M:%S"))
                        details.save()
                if att != 0:
                    person1 = User.objects.get(username=id)
                    if person != person1:
                        att = 0
            else:
                # Unknown Face
                id = "Who are you ?"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, predicted, (x + 150, y - 5), font, 1, (155, 0, 155), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        # Escape to exit the webcam / program
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            for i in list1:
                if i != 'engaged':
                    temp = True
                    break
            if temp:
                print(np.argmax(list1))
            break
    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()
    return redirect('home')


def logout_User(request):
    logout(request)
    return redirect('register')


def staff_home(request):
    return render(request, 'staff-home.html')


def total_attendance(request):
    engaged = 4
    confusion = 3
    bored = 2
    frustrated = 1
    total = 0
    count = 0
    engaged_count = 0
    confusion_count = 0
    bored_count = 0
    frustrated_count = 0
    var = last_login.objects.all().last()
    username = User.objects.get(username=var.username)
    print(username)
    vartime = class_hour.objects.filter(faculty=username)
    list1 = []
    list2 = attention.objects.all()
    for i in vartime:
        print(i)
        list1.append(attention.objects.filter(date=i.date))
    for i in vartime:
        print(i.start_time)
        print(i.end_time)
        for j in list2:
            if j.Time <= i.end_time or j.Time >= i.start_time:
                count += 1
                if j.emotion == 'engaged':
                    emotion = engaged
                    engaged_count += 1
                elif j.emotion == 'confusion':
                    emotion = confusion
                    confusion_count += 1
                elif j.emotion == 'bored':
                    emotion = bored
                    bored_count += 1
                else:
                    frustrated_count += 1
                    emotion = frustrated
                total += emotion
        print(total)
        print(count)
        print(round(total / count))
        em = ''
        if round(total / count) == 4:
            em = 'engaged'
        elif round(total / count) == 3:
            em = 'confusion'
        elif round(total / count) == 3:
            em = 'bored'
        else:
            em = 'frustrated'
        var1 = reports.objects.filter(class_hour=i)
        if len(var1)== 0:
            repvar = reports(class_hour=i, overview=round(total / count), review=em, engaged_count=engaged_count,
                         confusion_count=confusion_count, bored_count=bored_count, frustrated_count=frustrated_count)
            repvar.save()
    return render(request, 'total_attendance.html')


def view_review(request):
    list1 = []
    var = last_login.objects.all().last()
    username = User.objects.get(username=var.username)
    vartime = class_hour.objects.filter(faculty=username)
    for j in vartime:
        print(j)
        list1.append(reports.objects.filter(class_hour=j))
    print(len(list1))
    list2 = []
    for i in range(0,len(list1)):
        for k in list1[i]:
            list2.append(k)
    print(list2)
    return render(request, 'review.html',{'list1':list2})
