package org.pytorch.demo.nlp;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.InfoViewFactory;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;

import java.io.File;
import java.nio.charset.Charset;
import java.util.Locale;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;

public class TextClassificationActivity extends BaseModuleActivity {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";

  private static final long EDIT_TEXT_STOP_DELAY = 600l;
  private static final int TOP_N = 3;
  private static final String SCORES_FORMAT = "%.2f";

  private EditText mEditText;
  private View mResultContent;
  private ResultRowView[] mResultRowViews = new ResultRowView[3];

  private Module mModule;
  private String mModuleAssetName;

  private String mLastBgHandledText;
  private String[] mModuleClasses;

  private static class AnalysisResult {
    private final String[] topNClassNames;
    private final float[] topNScores;

    public AnalysisResult(String[] topNClassNames, float[] topNScores) {
      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
    }
  }

  private Runnable mOnEditTextStopRunnable = () -> {
    final String text = mEditText.getText().toString();
    mBackgroundHandler.post(() -> {
      if (TextUtils.equals(text, mLastBgHandledText)) {
        return;
      }

      if (TextUtils.isEmpty(text)) {
        runOnUiThread(() -> applyUIEmptyTextState());
        mLastBgHandledText = null;
        return;
      }

      final AnalysisResult result = analyzeText(text);
      if (result != null) {
        runOnUiThread(() -> applyUIAnalysisResult(result));
        mLastBgHandledText = text;
      }
    });
  };

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_text_classification);
    mEditText = findViewById(R.id.text_classification_edit_text);
    findViewById(R.id.text_classification_clear_button).setOnClickListener(v -> mEditText.setText(""));

    final ResultRowView headerRow = findViewById(R.id.text_classification_result_header_row);
    headerRow.nameTextView.setText(R.string.text_classification_topic);
    headerRow.scoreTextView.setText(R.string.text_classification_score);
    headerRow.setVisibility(View.VISIBLE);

    mResultRowViews[0] = findViewById(R.id.text_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.text_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.text_classification_top3_result_row);
    mResultContent = findViewById(R.id.text_classification_result_content);

    mEditText.addTextChangedListener(new InternalTextWatcher());
  }

  protected String getModuleAssetName() {
    if (!TextUtils.isEmpty(mModuleAssetName)) {
      return mModuleAssetName;
    }

    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
    mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
        ? moduleAssetNameFromIntent
        : "model-reddit16-f140225004_2.pt1";

    return mModuleAssetName;
  }

  @WorkerThread
  @Nullable
  private AnalysisResult analyzeText(final String text) {
    if (mModule == null) {
      final String moduleFileAbsoluteFilePath = new File(
          Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
      mModule = Module.load(moduleFileAbsoluteFilePath);

      final IValue getClassesOutput = mModule.runMethod("get_classes");

      final IValue[] classesListIValue = getClassesOutput.getList();
      final String[] moduleClasses = new String[classesListIValue.length];

      int i = 0;
      for (IValue iv : classesListIValue) {
        moduleClasses[i++] = iv.getString();
      }

      mModuleClasses = moduleClasses;
    }

    final long[] shape = new long[]{1, text.length()};
    byte[] bytes = text.getBytes(Charset.forName("UTF-8"));
    final Tensor inputTensor = Tensor.newUInt8Tensor(shape, bytes);

    final Tensor outputTensor = mModule.forward(IValue.tensor(inputTensor)).getTensor();
    final float[] scores = outputTensor.getDataAsFloatArray();
    final int[] ixs = Utils.getTopNIxs(scores, TOP_N);

    final String[] topNClassNames = new String[TOP_N];
    final float[] topNScores = new float[TOP_N];
    for (int i = 0; i < TOP_N; i++) {
      final int ix = ixs[i];
      topNClassNames[i] = mModuleClasses[ix];
      topNScores[i] = scores[ix];
    }

    return new AnalysisResult(topNClassNames, topNScores);
  }

  private void applyUIAnalysisResult(AnalysisResult result) {
    for (int i = 0; i < TOP_N; i++) {
      setUiResultRowView(
          mResultRowViews[i],
          result.topNClassNames[i],
          String.format(Locale.US, SCORES_FORMAT, result.topNScores[i]));
    }

    mResultContent.setVisibility(View.VISIBLE);
  }

  private void applyUIEmptyTextState() {
    mResultContent.setVisibility(View.GONE);
  }

  private void setUiResultRowView(ResultRowView resultRowView, String name, String score) {
    resultRowView.nameTextView.setText(name);
    resultRowView.scoreTextView.setText(score);
    resultRowView.setProgressState(false);
  }

  @Override
  protected int getInfoViewCode() {
    return InfoViewFactory.INFO_VIEW_TYPE_TEXT_CLASSIFICATION;
  }

  private class InternalTextWatcher implements TextWatcher {
    @Override
    public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

    @Override
    public void onTextChanged(CharSequence s, int start, int before, int count) {}

    @Override
    public void afterTextChanged(Editable s) {
      mUIHandler.removeCallbacks(mOnEditTextStopRunnable);
      mUIHandler.postDelayed(mOnEditTextStopRunnable, EDIT_TEXT_STOP_DELAY);
    }
  }

}
