package org.pytorch.demo.vision;

import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.view.TextureView;
import android.view.View;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.InfoViewFactory;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.util.Locale;

import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";

  private static final int INPUT_TENSOR_WIDTH = 224;
  private static final int INPUT_TENSOR_HEIGHT = 224;
  private static final int TOP_N = 3;
  private static final String FORMAT_MS = "%dms";
  private static final String FORMAT_FPS = "%.1fFPS";
  public static final String SCORES_FORMAT = "%.2f";

  private boolean mAnalyzeImageErrorState;

  static class AnalysisResult {

    private final String[] topNClassNames;
    private final float[] topNScores;
    private final long analysisDuration;

    public AnalysisResult(String[] topNClassNames, float[] topNScores, long analysisDuration) {
      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
      this.analysisDuration = analysisDuration;
    }
  }

  private ResultRowView[] mResultRowViews = new ResultRowView[TOP_N];
  private TextView mFpsText;
  private TextView mMsText;
  private Module mModule;
  private String mModuleAssetName;

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  @Override
  protected TextureView onCreateGetCameraPreviewTextureView() {
    return findViewById(R.id.image_classification_texture_view);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    final ResultRowView headerResultRowView =
        findViewById(R.id.image_classification_result_header_row);
    headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
    headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

    mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);

    mFpsText = findViewById(R.id.image_classification_fps_text);
    mMsText = findViewById(R.id.image_classification_ms_text);
  }

  @Override
  protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
    for (int i = 0; i < TOP_N; i++) {
      final ResultRowView rowView = mResultRowViews[i];
      rowView.nameTextView.setText(result.topNClassNames[i]);
      rowView.scoreTextView.setText(String.format(Locale.US, SCORES_FORMAT,
          result.topNScores[i]));
      rowView.setProgressState(false);
    }

    mMsText.setText(String.format(Locale.US, FORMAT_MS, result.analysisDuration));
    if (mMsText.getVisibility() != View.VISIBLE) {
      mMsText.setVisibility(View.VISIBLE);
    }
    mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
    if (mFpsText.getVisibility() != View.VISIBLE) {
      mFpsText.setVisibility(View.VISIBLE);
    }
  }

  protected String getModuleAssetName() {
    if (!TextUtils.isEmpty(mModuleAssetName)) {
      return mModuleAssetName;
    }

    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
    mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
        ? moduleAssetNameFromIntent
        : "resnet18.pt";

    return mModuleAssetName;
  }

  @Override
  protected String getInfoViewAdditionalText() {
    return getModuleAssetName();
  }

  @Override
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
        mModule = Module.load(moduleFileAbsoluteFilePath);
      }

      final long startTime = SystemClock.elapsedRealtime();
      final Tensor inputTensor =
          TensorImageUtils.imageYUV420CenterCropToFloatTensorTorchVisionForm(
              image.getImage(), rotationDegrees, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT);

      final Tensor outputTensor = mModule.forward(IValue.tensor(inputTensor)).getTensor();
      final float[] scores = outputTensor.getDataAsFloatArray();
      final int[] ixs = Utils.getTopNIxs(scores, TOP_N);

      final String[] topNClassNames = new String[TOP_N];
      final float[] topNScores = new float[TOP_N];
      for (int i = 0; i < TOP_N; i++) {
        final int ix = ixs[i];
        topNClassNames[i] = Constants.IMAGENET_CLASSES[ix];
        topNScores[i] = scores[ix];
      }
      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(topNClassNames, topNScores, analysisDuration);
    } catch (Exception e) {
      mAnalyzeImageErrorState = true;
      showErrorDialog(v -> ImageClassificationActivity.this.finish());
      return null;
    }
  }

  @Override
  protected int getInfoViewCode() {
    return InfoViewFactory.INFO_VIEW_TYPE_IMAGE_CLASSIFICATION;
  }
}
