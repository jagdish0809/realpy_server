from fastapi import FastAPI, Body, Header , HTTPException
import base64
from fastapi.responses import JSONResponse
import uvicorn
from insightface.app import FaceAnalysis
import cv2
import insightface
import numpy as np
import gc

app = FastAPI()

response_dict={
    "Response":"",
    "Image_Base64":""
}
user="Event"
password="RandomGeneratedPassword@"

@app.post("/extract_face")
async def extract_text_from_pdf(image_data: dict = Body(...), userid: str = Header(None), clientsecretkey: str = Header(None)):
    try:
        if userid is not None and clientsecretkey is not None:
            hdruserid = userid
            hdrclientsecretkey = clientsecretkey
        else:
            response_dict["Response"]="Failed. Invalid Header"
            response_dict["Image_Base64"]="Could not be generated"
            gc.collect()
            return JSONResponse(response_dict)
        if hdruserid==user and hdrclientsecretkey==password:

            image_name = image_data.get('Image_Name', None)
            image_base=image_data.get('Image_Base64', None)
            # Load the buffalo model for face detection
            model = FaceAnalysis(name='buffalo_l')
            model.prepare(ctx_id=0, det_size=(640, 640))
            
            # Decode base64 image and convert to numpy array
            decoded_image = base64.b64decode(image_base)
            np_arr = np.frombuffer(decoded_image, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Perform face detection
            faces = model.get(image)
            
            if len(faces) > 0:
                # Assuming there's only one face in the image
                face = faces[0]

                if face.det_score <=0.77:
                    response_dict["Response"]="Image quality is not good"
                    response_dict["Image_Base64"]="Could not be generated"
                    gc.collect()
                    return JSONResponse(response_dict)
                     
                
                # Extract the bounding box coordinates
                bbox = face.bbox.astype(int)
                
                # Calculate expansion percentage (50%)
                expansion_percentage = 0.5
                
                # Calculate expanded bounding box coordinates
                expanded_bbox = [
                    max(0, bbox[0] - int((bbox[2] - bbox[0]) * expansion_percentage)),
                    max(0, bbox[1] - int((bbox[3] - bbox[1]) * expansion_percentage)),
                    min(image.shape[1], bbox[2] + int((bbox[2] - bbox[0]) * expansion_percentage)),
                    min(image.shape[0], bbox[3] + int((bbox[3] - bbox[1]) * expansion_percentage))
                ]
                
                # Crop the expanded face image
                expanded_face = image[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                
                # Convert the expanded face image to base64
                _, buffer = cv2.imencode(".jpg", expanded_face)
                img_str = base64.b64encode(buffer).decode("utf-8")
                return img_str

            else:
                raise HTTPException(status_code=400, detail="No face detected in the image")
        
        else:
            response_dict["Response"]="Failed. Invalid Header"
            response_dict["Image_Base64"]="Could not be generated"
            gc.collect()
            return JSONResponse(response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="localhost", port=8000, reload=True)
