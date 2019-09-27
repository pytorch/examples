package org.pytorch.demo.nlp;

import android.content.Intent;
import android.os.Bundle;

import org.pytorch.demo.AbstractListActivity;
import org.pytorch.demo.R;
import org.pytorch.demo.vision.ImageClassificationActivity;
import org.pytorch.demo.vision.VisionListActivity;

public class NLPListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    findViewById(R.id.nlp_card_lstm_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(NLPListActivity.this, TextClassificationActivity.class);
      startActivity(intent);
    });
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.nlp_list_content;
  }
}
