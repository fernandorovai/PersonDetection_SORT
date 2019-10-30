import cv2


class Painter():
    def __init__(self):
        return

    @staticmethod
    def DrawBox(frame, coords, videoRes, color=(0, 0, 255), thickness=5):
        if coords is None:
            return frame

        trackID = ''
        width, height = videoRes

        xMin, yMin, xMax, yMax, labelID = coords

        if xMin < 0:
            xMin = 0
        if yMin < 0:
            yMin = 0
        if xMax < 0:
            xMax = 0
        if yMax < 0:
            yMax = 0

        xMin = xMin*width
        xMax = xMax*width
        yMin = yMin*height
        yMax = yMax*height

        try:
            cv2.rectangle(frame, (int(xMin), int(yMin)),
                          (int(xMax), int(yMax)), color, thickness)
        except Exception as e:
            print(e)
        return frame

    @staticmethod
    def DrawFPS(frame, fps):
        if frame is not None:
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            pos = (10, 20)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 1

            cv2.putText(frame, str(round(fps, 2)),
                        pos,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        return frame

    @staticmethod
    def DrawTotalBoxes(frame, numBoxes):
        if frame is not None:
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            pos = (10, 40)
            fontScale = 1
            fontColor = (0, 255, 0)
            lineType = 1

            cv2.putText(frame, str(numBoxes),
                        pos,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        return frame
