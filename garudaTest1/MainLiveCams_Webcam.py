def calculate_obj_coordinates(frame):
    try:
        results = model.predict(frame, verbose=False)
        object_coords = []
        for result in results:
            boxes = result.boxes
            for box in boxes:  # Process all boxes instead of just the first one
                x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
                x = (x1 + x2) // 2
                y = y2
                # Get class ID and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                object_coords.append((x, y, cls_id, conf))
        return object_coords
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []
