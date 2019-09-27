package org.pytorch.demo.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Size;
import android.view.TextureView;
import android.widget.Toast;

import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.StatusBarUtils;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;

public abstract class AbstractCameraXActivity<R> extends BaseModuleActivity {
  private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

  private TextureView mTextureView;

  protected abstract int getContentViewLayoutId();

  protected abstract TextureView onCreateGetCameraPreviewTextureView();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(getContentViewLayoutId());
    StatusBarUtils.setStatusBarOverlay(getWindow(), true);
    mTextureView = onCreateGetCameraPreviewTextureView();

    startBackgroundThread();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(
          this,
          PERMISSIONS,
          REQUEST_CODE_CAMERA_PERMISSION);
    } else {
      setupCameraX();
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(
            this,
            "You can't use image classification example without granting CAMERA permission",
            Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        setupCameraX();
      }
    }
  }

  private void setupCameraX() {
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(output -> mTextureView.setSurfaceTexture(output.getSurfaceTexture()));

    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(224, 224))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(
        (image, rotationDegrees) -> {
          final R result = analyzeImage(image, rotationDegrees);
          if (result != null) {
            runOnUiThread(() -> applyToUiAnalyzeImageResult(result));
          }
        });

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  @WorkerThread
  @Nullable
  protected abstract R analyzeImage(ImageProxy image, int rotationDegrees);

  @UiThread
  protected abstract void applyToUiAnalyzeImageResult(R result);
}
