import cv2

def selective_search_regions(img, mode="fast"):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()
    # rects: list of (x, y, w, h)
    rects = rects[:400]     # keep top-N proposals per image
    return rects
