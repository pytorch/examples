package org.pytorch.demo;

import android.view.View;
import android.view.Window;

import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class StatusBarUtils {
  public static void setStatusBarOverlay(Window window, final boolean showStatusBarAsOverlay) {
    View decorView = window.getDecorView();
    ViewCompat.setOnApplyWindowInsetsListener(
        decorView,
        (v, insets) -> {
          WindowInsetsCompat defaultInsets = ViewCompat.onApplyWindowInsets(v, insets);
          return defaultInsets.replaceSystemWindowInsets(
              defaultInsets.getSystemWindowInsetLeft(),
              showStatusBarAsOverlay ? 0 : defaultInsets.getSystemWindowInsetTop(),
              defaultInsets.getSystemWindowInsetRight(),
              defaultInsets.getSystemWindowInsetBottom());
        });
    ViewCompat.requestApplyInsets(decorView);
  }
}
