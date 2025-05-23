# Plot with cv2
cv2.imshow("Custom Colors", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Delete repeated classes.
if len(class_ids) > 5:
    boxes = boxes[:5]
    class_ids = class_ids[:5]
    confidences = confidences[:5]
    classNames = classNames[:5]
for name, conf in zip(classNames, confidences):
    print(f"{name:<6} | Confidence: {conf:.2f}")