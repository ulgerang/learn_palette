from PIL import Image


def readPalleteColor( filePath):
    im = Image.open(filePath)
    colors =[]
    preP=None
    for x in range(0,im.width):


        p= im.getpixel( (x,0) )
        if(preP==p):
            break

        if preP!=None :
            colors.append(preP[0:3])
        preP = p
    return colors
