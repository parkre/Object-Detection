package com.example.jacobparker.goat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
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
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jacobparker on 2/28/16.
 */
public class ImageObjectDetectorActivity extends Activity
{
    private static final String TAG = "IMAGEOBJECTDETECTOR";

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

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                    if (objIndex == 3)
                    {
                        detectMultipleObjects();
                    }
                    else
                    {
                        detectObject();
                    }
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
        setContentView(R.layout.image_object_detector_activity);

        objIndex = getIntent().getIntExtra("objIndex", 0);
        envIndex = getIntent().getIntExtra("envIndex", 0);

        if (objIndex < 3)
        {
            objId = objResIDs[objIndex];
        }

        envId = envResIDs[envIndex];

        if (!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        }
        else
        {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void detectMultipleObjects()
    {
        try
        {
            env_mat = new Mat();
            b = BitmapFactory.decodeResource(getResources(), envId);
            Utils.bitmapToMat(b, env_mat);
        }
        catch(Exception e)
        {
            // logic
        }

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);

        MatOfKeyPoint env_keypoints_mat = new MatOfKeyPoint();

        startTime = System.currentTimeMillis();
        detector.detect(env_mat, env_keypoints_mat);
        stopTime = System.currentTimeMillis();
        detectionTime += (stopTime - startTime);

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        Mat env_descriptor = new Mat();

        startTime = System.currentTimeMillis();
        extractor.compute(env_mat, env_keypoints_mat, env_descriptor);
        stopTime = System.currentTimeMillis();
        descriptionTime += (stopTime - startTime);

        for (int i = 0; i < 3; i++)
        {
            /* read obj and env images */
            try
            {
                obj_mat = new Mat();
                b = BitmapFactory.decodeResource(getResources(), objResIDs[i]);
                Utils.bitmapToMat(b, obj_mat);
            }
            catch (Exception e)
            {
                // logic
            }

            /* detect keypoints in object image */
            MatOfKeyPoint obj_keypoints_mat = new MatOfKeyPoint();
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

            List<MatOfDMatch> matches = new ArrayList();

            List<KeyPoint> env_keypoints_list;
            Mat obj_descriptor = new Mat();

            startTime = System.currentTimeMillis();
            detector.detect(obj_mat, obj_keypoints_mat);
            stopTime = System.currentTimeMillis();
            detectionTime += (stopTime - startTime);

            startTime = System.currentTimeMillis();
            extractor.compute(obj_mat, obj_keypoints_mat, obj_descriptor);
            stopTime = System.currentTimeMillis();
            descriptionTime += (stopTime - startTime);

            startTime = System.currentTimeMillis();
            matcher.knnMatch(obj_descriptor, env_descriptor, matches, 2);
            stopTime = System.currentTimeMillis();
            matchingTime += (stopTime - startTime);

            env_keypoints_list = env_keypoints_mat.toList();

            /* filter out matches based on hamming distance */
            LinkedList<DMatch> good_matches = new LinkedList();
            for (int k = 0; k < matches.size(); k++)
            {
                List<DMatch> dm_list = matches.get(k).toList();
                if (dm_list.get(0).distance / dm_list.get(1).distance < scales[i])
                {
                    good_matches.addLast(dm_list.get(0));
                }
            }

            if (good_matches.size() < 10)
            {
                // no matches
                continue;
            }

            List<KeyPoint> obj_keypoints_list = obj_keypoints_mat.toList();

            LinkedList<Point> obj_kp_llist = new LinkedList<Point>();
            LinkedList<Point> env_kp_llist = new LinkedList<Point>();

            for (int k = 0; k < good_matches.size(); k++)
            {
                obj_kp_llist.addLast(obj_keypoints_list.get(good_matches.get(k).queryIdx).pt);
                env_kp_llist.addLast(env_keypoints_list.get(good_matches.get(k).trainIdx).pt);
            }

            MatOfPoint2f obj = new MatOfPoint2f();
            obj.fromList(obj_kp_llist);

            MatOfPoint2f env = new MatOfPoint2f();
            env.fromList(env_kp_llist);

            startTime = System.currentTimeMillis();
            Mat hg = Calib3d.findHomography(obj, env, Calib3d.RANSAC, Calib3d.FM_RANSAC);
            stopTime = System.currentTimeMillis();
            hgTime += (stopTime - startTime);

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
        }

        imgView = (ImageView) findViewById(R.id.img_view);

        try
        {
            Bitmap bmp = Bitmap.createBitmap(env_mat.cols(), env_mat.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(env_mat, bmp);
            imgView.setImageBitmap(bmp);
        }
        catch (CvException e)
        {
            Log.d("MAT_TO_BITMAP_EXCEPTION", e.getMessage());
        }
        Log.d(TAG, "FEATURE DETECTION="+detectionTime);
        Log.d(TAG, "FEATURE DESCRIPTION="+descriptionTime);
        Log.d(TAG, "DESCRIPTION MATCHING="+matchingTime);
        Log.d(TAG, "HGTIME="+hgTime);
    }

    public void detectObject()
    {
        try
        {
            env_mat = Utils.loadResource(this, envId, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        }
        catch(IOException e)
        {
            // logic
        }

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);

        MatOfKeyPoint env_keypoints_mat = new MatOfKeyPoint();
        detector.detect(env_mat, env_keypoints_mat);

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        Mat env_descriptor = new Mat();
        extractor.compute(env_mat, env_keypoints_mat, env_descriptor);

        try
        {
            obj_mat = Utils.loadResource(this, objResIDs[objIndex], Imgcodecs.CV_LOAD_IMAGE_COLOR);
        }
        catch (IOException e)
        {
            // logic
        }

        /* detect keypoints in object image */
        MatOfKeyPoint obj_keypoints_mat = new MatOfKeyPoint();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        List<MatOfDMatch> matches = new ArrayList();

        List<KeyPoint> env_keypoints_list;
        Mat obj_descriptor = new Mat();
        detector.detect(obj_mat, obj_keypoints_mat);
        extractor.compute(obj_mat, obj_keypoints_mat, obj_descriptor);
        matcher.knnMatch(obj_descriptor, env_descriptor, matches, 2);
        env_keypoints_list = env_keypoints_mat.toList();

        /* filter out matches based on hamming distance */
        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();

        for (int k = 0; k < matches.size(); k++)
        {
            List<DMatch> dm_list = matches.get(k).toList();
            if (dm_list.get(0).distance / dm_list.get(1).distance < .7)
            {
                good_matches.addLast(dm_list.get(0));
            }
        }

        if (good_matches.size() < 10)
        {
            Toast.makeText(this, "Unable to detect object", Toast.LENGTH_LONG).show();
            return;
        }

        List<KeyPoint> obj_keypoints_list = obj_keypoints_mat.toList();

        LinkedList<Point> obj_kp_llist = new LinkedList<Point>();
        LinkedList<Point> env_kp_llist = new LinkedList<Point>();

        for (int k = 0; k < good_matches.size(); k++)
        {
            obj_kp_llist.addLast(obj_keypoints_list.get(good_matches.get(k).queryIdx).pt);
            env_kp_llist.addLast(env_keypoints_list.get(good_matches.get(k).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(obj_kp_llist);

        MatOfPoint2f env = new MatOfPoint2f();
        env.fromList(env_kp_llist);

        startTime = System.currentTimeMillis();
        Mat hg = Calib3d.findHomography(obj, env, Calib3d.RANSAC, Calib3d.FM_RANSAC);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
        Log.d(TAG, "RANSAC_HOMOGRAPHY: " + elapsedTime);

        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2);
        Mat envCorners = new Mat(4, 1, CvType.CV_32FC2);

        objCorners.put(0, 0, new double[]{0, 0});
        objCorners.put(1, 0, new double[]{obj_mat.cols(), 0});
        objCorners.put(2, 0, new double[]{obj_mat.cols(), obj_mat.rows()});
        objCorners.put(3, 0, new double[]{0, obj_mat.rows()});

        startTime = System.currentTimeMillis();
        Core.perspectiveTransform(objCorners, envCorners, hg);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
        Log.d(TAG, "PERSPECTIVETRANSFORM: " + elapsedTime);

        Imgproc.line(env_mat, new Point(envCorners.get(0, 0)), new Point(envCorners.get(1, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(env_mat, new Point(envCorners.get(1, 0)), new Point(envCorners.get(2, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(env_mat, new Point(envCorners.get(2, 0)), new Point(envCorners.get(3, 0)), new Scalar(0, 255, 0), 4);
        Imgproc.line(env_mat, new Point(envCorners.get(3, 0)), new Point(envCorners.get(0, 0)), new Scalar(0, 255, 0), 4);

        imgView = (ImageView) findViewById(R.id.img_view);

        try
        {
            Imgproc.cvtColor(env_mat, env_mat, Imgproc.COLOR_BGR2RGB);
            Bitmap bmp = Bitmap.createBitmap(env_mat.cols(), env_mat.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(env_mat, bmp);
            imgView.setImageBitmap(bmp);
        }
        catch (CvException e)
        {
            Log.d("MAT_TO_BITMAP_EXCEPTION", e.getMessage());
        }
    }
}
