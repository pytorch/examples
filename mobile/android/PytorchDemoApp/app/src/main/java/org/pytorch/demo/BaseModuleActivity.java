package org.pytorch.demo;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

public class BaseModuleActivity extends AppCompatActivity {
  private static final int UNSET = 0;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  protected Handler mUIHandler;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    mUIHandler = new Handler(getMainLooper());
  }

  @Override
  protected void onPostCreate(@Nullable Bundle savedInstanceState) {
    super.onPostCreate(savedInstanceState);
    final Toolbar toolbar = findViewById(R.id.toolbar);
    if (toolbar != null) {
      setSupportActionBar(toolbar);
    }
    startBackgroundThread();
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread("ModuleActivity");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e(Constants.TAG, "Error on stopping background thread", e);
    }
  }

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    getMenuInflater().inflate(R.menu.menu_model, menu);
    menu.findItem(R.id.action_info).setVisible(getInfoViewCode() != UNSET);
    return true;
  }

  @Override
  public boolean onOptionsItemSelected(MenuItem item) {
    if (item.getItemId() == R.id.action_info) {
      onMenuItemInfoSelected();
    }
    return super.onOptionsItemSelected(item);
  }

  protected int getInfoViewCode() {
    return UNSET;
  }

  protected String getInfoViewAdditionalText() {
    return null;
  }

  private void onMenuItemInfoSelected() {
    final AlertDialog.Builder builder = new AlertDialog.Builder(this)
        .setCancelable(true)
        .setView(InfoViewFactory.newInfoView(this, getInfoViewCode(), getInfoViewAdditionalText()));

    builder.show();
  }

  protected void showErrorDialog(View.OnClickListener clickListener) {
    final View view = InfoViewFactory.newErrorDialogView(this);
    view.setOnClickListener(clickListener);
    final AlertDialog.Builder builder = new AlertDialog.Builder(this, R.style.CustomDialog)
        .setCancelable(false)
        .setView(view);
    builder.show();
  }
}
