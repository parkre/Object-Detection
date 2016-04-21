package com.example.jacobparker.goat;

import android.app.Activity;
import android.content.Intent;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class MainActivity extends Activity
{
    private static final String TAG = "MAINACTIVITY";

    private ArrayList<String> objNames = new ArrayList(Arrays.asList("book", "sanitizer", "router", "all objects"));
    private ArrayList<String> envNames = new ArrayList(Arrays.asList("test1", "test2", "multitest"));

    private int objIndex;
    private int envIndex;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Spinner objSpinner = (Spinner) findViewById(R.id.obj_spinner);
        Spinner envSpinner = (Spinner) findViewById(R.id.env_spinner);

        // Create an ArrayAdapter using the string array and a default spinner layout
        ArrayAdapter<String> objAdapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_item, android.R.id.text1, objNames);
        ArrayAdapter<String> envAdapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_item, android.R.id.text1, envNames);

        // Specify the layout to use when the list of choices appears
        objAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        envAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        // Apply the adapter to the spinner
        objSpinner.setAdapter(objAdapter);
        envSpinner.setAdapter(envAdapter);
        objSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener()
        {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id)
            {
                objIndex = pos;
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent)
            {
                // Another interface callback
            }
        });
        envSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener()
        {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id)
            {
                envIndex = pos;
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent)
            {
                // no selection interface callback
            }
        });
    }

    public void searchImage(View v)
    {
        Intent intent = new Intent(this, ImageObjectDetectorActivity.class);
        intent.putExtra("objIndex", objIndex);
        intent.putExtra("envIndex", envIndex);
        startActivity(intent);
    }

    public void searchRealTime(View v)
    {
        Log.d(TAG, "SEARCHREALTIME");
        Intent intent = new Intent(this, RealTimeObjectDetectorActivity.class);
        intent.putExtra("obj", objIndex);
        startActivity(intent);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu)
    {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item)
    {
        int id = item.getItemId();

        if (id == R.id.action_settings)
        {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
