package org.pytorch.demo;

import android.content.Intent;
import android.os.Bundle;

import org.pytorch.demo.nlp.NLPListActivity;
import org.pytorch.demo.vision.VisionListActivity;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    findViewById(R.id.main_vision_click_view).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, VisionListActivity.class)));
    findViewById(R.id.main_nlp_click_view).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, NLPListActivity.class)));
  }
}
