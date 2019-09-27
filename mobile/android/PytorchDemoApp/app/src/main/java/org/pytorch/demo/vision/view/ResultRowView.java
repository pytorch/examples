package org.pytorch.demo.vision.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.pytorch.demo.R;

import androidx.annotation.DimenRes;
import androidx.annotation.DrawableRes;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.Px;
import androidx.annotation.StyleRes;

public class ResultRowView extends RelativeLayout {

  public final TextView nameTextView;
  public final TextView scoreTextView;
  private final @Px
  int mProgressBarHeightPx;
  private final @Px
  int mProgressBarPaddingPx;
  @Nullable
  private final Drawable mProgressBarDrawable;
  @Nullable
  private final Drawable mProgressBarProgressStateDrawable;

  private boolean mIsInProgress = true;

  public ResultRowView(@NonNull Context context) {
    this(context, null);
  }

  public ResultRowView(@NonNull Context context, @Nullable AttributeSet attrs) {
    this(context, attrs, 0);
  }

  public ResultRowView(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
    this(context, attrs, defStyleAttr, 0);
  }

  public ResultRowView(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr,
                       int defStyleRes) {
    super(context, attrs, defStyleAttr, defStyleRes);
    inflate(context, R.layout.image_classification_result_row, this);
    nameTextView = findViewById(R.id.result_row_name_text);
    scoreTextView = findViewById(R.id.result_row_score_text);

    TypedArray a = context.getTheme().obtainStyledAttributes(
        attrs,
        R.styleable.ResultRowView,
        defStyleAttr, defStyleRes);
    try {
      final @StyleRes int textAppearanceResId =
          a.getResourceId(R.styleable.ResultRowView_textAppearance,
              R.style.TextAppearanceImageClassificationResultTop2Plus);

      nameTextView.setTextAppearance(context, textAppearanceResId);
      scoreTextView.setTextAppearance(context, textAppearanceResId);

      final @DimenRes int progressBarHeightDimenResId =
          a.getResourceId(R.styleable.ResultRowView_progressBarHeightRes, 0);
      mProgressBarHeightPx = progressBarHeightDimenResId != 0
          ? getResources().getDimensionPixelSize(progressBarHeightDimenResId)
          : 0;

      final @DimenRes int progressBarPaddingDimenResId =
          a.getResourceId(R.styleable.ResultRowView_progressBarPaddingRes, 0);
      mProgressBarPaddingPx = progressBarPaddingDimenResId != 0
          ? getResources().getDimensionPixelSize(progressBarPaddingDimenResId)
          : 0;

      setPadding(getPaddingLeft(), getPaddingTop(), getPaddingRight(),
          getBottom() + mProgressBarPaddingPx + mProgressBarHeightPx);

      final @DrawableRes int progressBarDrawableResId =
          a.getResourceId(R.styleable.ResultRowView_progressBarDrawableRes, 0);
      mProgressBarDrawable = progressBarDrawableResId != 0
          ? getResources().getDrawable(progressBarDrawableResId, null)
          : null;

      final @DrawableRes int progressBarDrawableProgressStateResId =
          a.getResourceId(R.styleable.ResultRowView_progressBarDrawableProgressStateRes, 0);
      mProgressBarProgressStateDrawable =
          progressBarDrawableResId != 0
              ? getResources().getDrawable(progressBarDrawableProgressStateResId, null)
              : null;
    } finally {
      a.recycle();
    }
  }

  @Override
  protected void dispatchDraw(Canvas canvas) {
    super.dispatchDraw(canvas);
    final Drawable drawable = mIsInProgress ? mProgressBarProgressStateDrawable :
        mProgressBarDrawable;

    if (drawable != null) {
      final int h = canvas.getHeight();
      final int w = canvas.getWidth();
      drawable.setBounds(0, h - mProgressBarHeightPx, w, h);
      drawable.draw(canvas);
    }
  }

  public void setProgressState(boolean isInProgress) {
    final boolean changed = isInProgress != mIsInProgress;
    mIsInProgress = isInProgress;
    if (isInProgress) {
      nameTextView.setVisibility(View.INVISIBLE);
      scoreTextView.setVisibility(View.INVISIBLE);
    } else {
      nameTextView.setVisibility(View.VISIBLE);
      scoreTextView.setVisibility(View.VISIBLE);
    }

    if (changed) {
      invalidate();
    }
  }
}
