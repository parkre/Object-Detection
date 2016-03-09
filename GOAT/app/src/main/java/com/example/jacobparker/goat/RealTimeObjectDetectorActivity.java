package com.example.jacobparker.goat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;

import android.app.Activity;
import android.content.ContextWrapper;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by jacobparker on 2/25/16.
 */
public class RealTimeObjectDetectorActivity extends Activity implements CvCameraViewListener2, View.OnTouchListener
{
    private static final String TAG = "REALTIMEOBJECTDETECTOR";

    //private CameraBridgeViewBase camViewer;
    private JavaCameraView camViewer;

    private Mat imgFrame;
    private Mat imgObject;
    //private Mat mGrey;

    private int mAbsoluteObjectSize = 0;

    static {
        if (!OpenCVLoader.initDebug())
            Log.e(TAG, "Failed to load OpenCV");
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                    Log.d(TAG, "OPENCV LOADED SUCCESSFULLY");
                    camViewer.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.realtime_object_detector_activity);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        camViewer = (JavaCameraView) findViewById(R.id.surface_view);
        camViewer.setVisibility(SurfaceView.VISIBLE);
        camViewer.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (camViewer != null)
        {
            camViewer.disableView();
        }
    }

    @Override
    public void onResume()
    {
        super.onResume();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        if (camViewer != null)
        {
            camViewer.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height)
    {
        imgObject = Imgcodecs.imread("res/images/obj2");
        imgFrame = new Mat();
        //mGrey = new Mat();
    }

    @Override
    public void onCameraViewStopped()
    {
        imgFrame.release();
        //mGrey.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        imgFrame = inputFrame.rgba();
        //mGrey = inputFrame.gray();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);

        MatOfKeyPoint keypointsObject = new MatOfKeyPoint();
        MatOfKeyPoint keypointsScene = new MatOfKeyPoint();

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        Mat descriptorObject = new Mat();
        Mat descriptorScene = new Mat();

        extractor.compute(imgObject, keypointsObject, descriptorObject);

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        MatOfDMatch matches = new MatOfDMatch();

        List<DMatch> matchesList = matches.toList();

        Double maxDist = 0.0;
        Double minDist = 100.0;

        for (int i = 0; i < descriptorObject.rows(); i++)
        {
            Double dist = (double) matchesList.get(i).distance;
            if (dist < minDist)
            {
                minDist = dist;
            }
            if (dist > maxDist)
            {
                maxDist = dist;
            }
        }

        LinkedList<DMatch> goodMatches = new LinkedList();
        MatOfDMatch gmMat = new MatOfDMatch();

        for (int i = 0; i < descriptorObject.rows(); i++)
        {
            if (matchesList.get(i).distance < 3 * minDist)
            {
                goodMatches.addLast(matchesList.get(i));
            }
        }

        gmMat.fromList(goodMatches);

        return inputFrame.rgba();
    }

    @Override
    public boolean onTouch(View v, MotionEvent mEvent)
    {
        Log.d(TAG, "ONTOUCH");
        return true;
    }


}
