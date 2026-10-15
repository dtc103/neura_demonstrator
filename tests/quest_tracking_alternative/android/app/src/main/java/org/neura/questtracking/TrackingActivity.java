package org.neura.questtracking;

import android.app.NativeActivity;
import android.os.Bundle;
import android.view.WindowManager;

public final class TrackingActivity extends NativeActivity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }
}
