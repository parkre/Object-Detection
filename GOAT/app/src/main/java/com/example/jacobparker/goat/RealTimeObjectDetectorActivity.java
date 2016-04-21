package com.example.jacobparker.goat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jacobparker on 2/25/16.
 */
public class RealTimeObjectDetectorActivity extends Activity implements CvCameraViewListener2, View.OnTouchListener
{
    private static final String TAG = "REALTIMEOBJECTDETECTOR";

    private CameraBridgeViewBase camViewer;
    //private JavaCameraView camViewer;

    private Integer [] objResIDs = { R.drawable.obj1, R.drawable.obj2, R.drawable.obj3 };
    private Integer [] envResIDs = { R.drawable.single_test1, R.drawable.single_test2, R.drawable.multi_test };
    private String [] names = { "hand sanitizer", "book", "router" };
    private double [] scales = { .95, .8, .8 };

    private int objIndex;
    private int envIndex;
    private Integer objId;
    private Integer envId;

    private Mat obj_mat;
    private Mat env_mat;

    private ImageView imgView;

    private long startTime;
    private long stopTime;
    private long elapsedTime;

    private long hgTime = 0;
    private long descriptionTime = 0;
    private long detectionTime = 0;
    private long matchingTime = 0;

    private Bitmap b;

    private int mAbsoluteObjectSize = 0;

    /* init for object mat */
    FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
    DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

    MatOfKeyPoint obj_keypoints_mat = new MatOfKeyPoint();
    DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

    Mat obj_descriptor;

    static
    {
        if (!OpenCVLoader.initDebug())
            Log.e(TAG, "Failed to load OpenCV");
    }

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
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
        super.onCreate(savedInstanceState);
        setContentView(R.layout.realtime_object_detector_activity);

        objIndex = getIntent().getIntExtra("objIndex", 0);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        camViewer = (JavaCameraView) findViewById(R.id.surface_view);
        camViewer.setVisibility(SurfaceView.VISIBLE);
        camViewer.setCvCameraViewListener(this);

        if (!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        }
        else
        {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
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
        obj_mat = Imgcodecs.imread("res/images/obj2");
        env_mat = new Mat();
    }

    @Override
    public void onCameraViewStopped()
    {
        env_mat.release();
    }

    public void init()
    {
        detector.detect(obj_mat, obj_keypoints_mat);
        extractor.compute(obj_mat, obj_keypoints_mat, obj_descriptor);
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        Log.d("ONCAMERAFRAME", "inside method");

        env_mat = inputFrame.rgba();

        MatOfKeyPoint env_keypoints_mat = new MatOfKeyPoint();
        detector.detect(env_mat, env_keypoints_mat);

        Mat env_descriptor = new Mat();

        extractor.compute(env_mat, env_keypoints_mat, env_descriptor);

        /* for loop placement for real time multiple object detection */

        List<MatOfDMatch> matches = new ArrayList();

        List<KeyPoint> env_keypoints_list;
        obj_descriptor = new Mat();
        detector.detect(obj_mat, obj_keypoints_mat);
        extractor.compute(obj_mat, obj_keypoints_mat, obj_descriptor);
        matcher.knnMatch(obj_descriptor, env_descriptor, matches, 2);

        env_keypoints_list = env_keypoints_mat.toList();

        /* filter out matches based on hamming distance */
        LinkedList<DMatch> good_matches = new LinkedList();
        for (int k = 0; k < matches.size(); k++)
        {
            List<DMatch> dm_list = matches.get(k).toList();
            if (dm_list.get(0).distance / dm_list.get(1).distance < scales[1])
            {
                good_matches.addLast(dm_list.get(0));
            }
        }

        if (good_matches.size() < 10)
        {
            // no matches
            return env_mat;
        }

        List<KeyPoint> obj_keypoints_list = obj_keypoints_mat.toList();

        LinkedList<Point> obj_kp_llist = new LinkedList();
        LinkedList<Point> env_kp_llist = new LinkedList();

        for (int k = 0; k < good_matches.size(); k++)
        {
            obj_kp_llist.addLast(obj_keypoints_list.get(good_matches.get(k).queryIdx).pt);
            env_kp_llist.addLast(env_keypoints_list.get(good_matches.get(k).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(obj_kp_llist);

        MatOfPoint2f env = new MatOfPoint2f();
        env.fromList(env_kp_llist);

        Mat hg = Calib3d.findHomography(obj, env, Calib3d.RANSAC, Calib3d.FM_RANSAC);

        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
        Mat envCorners = new Mat(4, 1, CvType.CV_32FC2);

        objCorners.put(0, 0, new double[]{0, 0});
        objCorners.put(1, 0, new double[]{obj_mat.cols(), 0});
        objCorners.put(2, 0, new double[]{obj_mat.cols(), obj_mat.rows()});
        objCorners.put(3, 0, new double[]{0, obj_mat.rows()});

        Core.perspectiveTransform(objCorners, envCorners, hg);

        Imgproc.line(env_mat, new Point(envCorners.get(0, 0)), new Point(envCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(env_mat, new Point(envCorners.get(1, 0)), new Point(envCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(env_mat, new Point(envCorners.get(2, 0)), new Point(envCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(env_mat, new Point(envCorners.get(3, 0)), new Point(envCorners.get(0, 0)), new Scalar(0, 255, 0), 4);

        return env_mat;
    }

    @Override
    public boolean onTouch(View v, MotionEvent mEvent)
    {
        return true;
    }
}
