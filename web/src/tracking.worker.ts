import {FilesetResolver,PoseLandmarker,FaceLandmarker,HandLandmarker} from '@mediapipe/tasks-vision';
let pose:PoseLandmarker,face:FaceLandmarker,hands:HandLandmarker;
self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      // MediaPipe clears ModuleFactory after each task. ESM imports are cached,
      // so explicitly restore its default export for the second/third task.
      (self as any).import=async(url:string)=>{const module=await import(/* @vite-ignore */ url);(self as any).ModuleFactory=module.default;};
      const files=await FilesetResolver.forVisionTasks('/models/wasm',true);
      pose=await PoseLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/pose_landmarker_lite.task',delegate:'CPU'},runningMode:'VIDEO',numPoses:1});
      face=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/face_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
      hands=await HandLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/hand_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numHands:2});
      self.postMessage({type:'ready'});
    }else if(data.type==='frame'){
      const t=performance.now(),body=pose.detectForVideo(data.bitmap,data.time),f=face.detectForVideo(data.bitmap,data.time),h=hands.detectForVideo(data.bitmap,data.time);
      data.bitmap.close();
      self.postMessage({type:'signals',body:body.worldLandmarks[0]??[],face:f.faceBlendshapes[0]?.categories??[],faceMatrix:f.facialTransformationMatrixes[0]?.data,hands:h.landmarks,handedness:h.handedness,ms:performance.now()-t});
    }
  }catch(error){data.bitmap?.close();self.postMessage({type:'error',message:String(error)});}
};
