package org.pytorch.demo;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.DrawableRes;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.StringRes;
import androidx.constraintlayout.widget.ConstraintLayout;

public class ListCardView extends ConstraintLayout {

  private final TextView mTitleTextView;
  private final TextView mDescriptionTextView;
  private final ImageView mImageView;

  public ListCardView(@NonNull Context context) {
    this(context, null);
  }

  public ListCardView(@NonNull Context context, @Nullable AttributeSet attrs) {
    this(context, attrs, 0);
  }

  public ListCardView(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
    super(context, attrs, defStyleAttr);
    inflate(context, R.layout.list_card, this);

    mTitleTextView = findViewById(R.id.list_card_title);
    mDescriptionTextView = findViewById(R.id.list_card_description);
    mImageView = findViewById(R.id.list_card_image);

    TypedArray a = context.getTheme().obtainStyledAttributes(
        attrs,
        R.styleable.ListCardView,
        defStyleAttr, 0);

    try {
      final @StringRes int titleResId = a.getResourceId(R.styleable.ListCardView_titleRes, 0);
      if (titleResId != 0) {
        mTitleTextView.setText(titleResId);
        mTitleTextView.setVisibility(View.VISIBLE);
      } else {
        mTitleTextView.setVisibility(View.GONE);
      }

      final @StringRes int descResId = a.getResourceId(R.styleable.ListCardView_descriptionRes, 0);
      if (descResId != 0) {
        mDescriptionTextView.setText(descResId);
        mDescriptionTextView.setVisibility(View.VISIBLE);
      } else {
        mDescriptionTextView.setVisibility(View.GONE);
      }

      final @DrawableRes int imageResId = a.getResourceId(R.styleable.ListCardView_imageRes, 0);
      if (imageResId != 0) {
        mImageView.setImageResource(imageResId);
        mImageView.setVisibility(View.VISIBLE);
      } else {
        mImageView.setVisibility(View.GONE);
      }
    } finally {
      a.recycle();
    }

  }
}
