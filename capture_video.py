import numpy as np
import cv2 as cv
import imagingcontrol4 as ic4

# serial_num = "42520778"
# ic4.Library.init()
# devices = ic4.DeviceEnum.devices()

# for dev in devices:
#     if dev.serial == serial_num:
#         print(f"Device found: {dev}")
#         print(f"Opening device with serial number: {serial_num}")
#         print(f"Device index: {devices.index(dev)}")
#         index = devices.index(dev)

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
while(True):
    # Capture image par imaghe
    ret, img = cap.read()
    # Préparation de l'affichage de l'image
    cv.imshow('frame',img)
    # affichage et saisie d'un code clavier
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# Ne pas oublier de fermer le flux et la fenetre
cap.release()
cv.destroyAllWindows()