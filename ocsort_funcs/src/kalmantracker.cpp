
// The code is based on:
// https://github.com/CnybTseng/JDE/blob/master/platforms/common/trajectory.cpp
// Ths copyright of CnybTseng/JDE is as follows:
// MIT License

#include <algorithm>
#include "kalmantracker.h"

void TKalmanFilter::init(const cv::Mat &measurement) {
  measurement.copyTo(statePost(cv::Rect(0, 0, 1, 4)));
  statePost(cv::Rect(0, 4, 1, 4)).setTo(0);
  statePost.copyTo(statePre);

  float varpos = std_weight_position * (*measurement.ptr<float>(3));
  varpos *= varpos;
  float varvel = 100 * std_weight_velocity * (*measurement.ptr<float>(3));
  varvel *= varvel;

  errorCovPost.setTo(0);
  *errorCovPost.ptr<float>(0, 0) = varpos;
  *errorCovPost.ptr<float>(1, 1) = varpos;
  *errorCovPost.ptr<float>(2, 2) = 1e-4f;
  *errorCovPost.ptr<float>(3, 3) = varpos;
  *errorCovPost.ptr<float>(4, 4) = varvel;
  *errorCovPost.ptr<float>(5, 5) = varvel;
  *errorCovPost.ptr<float>(6, 6) = 1e-10f;
  *errorCovPost.ptr<float>(7, 7) = varvel;
  errorCovPost.copyTo(errorCovPre);
}

const cv::Mat &TKalmanFilter::predict() {
  float varpos = std_weight_position * (*statePre.ptr<float>(3));
  varpos *= varpos;
  float varvel = 100 * std_weight_velocity * (*statePre.ptr<float>(3));
  varvel *= varvel;

  processNoiseCov.setTo(0);
  *processNoiseCov.ptr<float>(0, 0) = varpos;
  *processNoiseCov.ptr<float>(1, 1) = varpos;
  *processNoiseCov.ptr<float>(2, 2) = 1e-4f;
  *processNoiseCov.ptr<float>(3, 3) = varpos;
  *processNoiseCov.ptr<float>(4, 4) = varvel;
  *processNoiseCov.ptr<float>(5, 5) = varvel;
  *processNoiseCov.ptr<float>(6, 6) = 1e-10f;
  *processNoiseCov.ptr<float>(7, 7) = varvel;

  return cv::KalmanFilter::predict();
}

const cv::Mat &TKalmanFilter::correct(const cv::Mat &measurement) {
  float varpos = std_weight_position * (*measurement.ptr<float>(3));
  varpos *= varpos;

  measurementNoiseCov.setTo(0);
  *measurementNoiseCov.ptr<float>(0, 0) = varpos;
  *measurementNoiseCov.ptr<float>(1, 1) = varpos;
  *measurementNoiseCov.ptr<float>(2, 2) = 0;
  *measurementNoiseCov.ptr<float>(3, 3) = varpos;

  return cv::KalmanFilter::correct(measurement);
}

void TKalmanFilter::project(cv::Mat *mean, cv::Mat *covariance) const {
  float varpos = std_weight_position * (*statePost.ptr<float>(3));
  varpos *= varpos;

  cv::Mat measurementNoiseCov_ = cv::Mat::eye(4, 4, CV_32F);
  *measurementNoiseCov_.ptr<float>(0, 0) = varpos;
  *measurementNoiseCov_.ptr<float>(1, 1) = varpos;
  *measurementNoiseCov_.ptr<float>(2, 2) = 0;
  *measurementNoiseCov_.ptr<float>(3, 3) = varpos;

  *mean = measurementMatrix * statePost;
  cv::Mat temp = measurementMatrix * errorCovPost;
  gemm(temp,
       measurementMatrix,
       1,
       measurementNoiseCov_,
       1,
       *covariance,
       cv::GEMM_2_T);
}

int KalmanTracker::count = 0;

const cv::Mat &KalmanTracker::predict(void) {
  this->age += 1;
  if(this->time_since_update > 0) {
    this->hit_streak = 0;
  }
  this->time_since_update += 1;
  // if (state != Tracked) {
  //   *cv::KalmanFilter::statePost.ptr<float>(4) = 0;
  //   *cv::KalmanFilter::statePost.ptr<float>(5) = 0;
  //   *cv::KalmanFilter::statePost.ptr<float>(6) = 0;
  //   *cv::KalmanFilter::statePost.ptr<float>(7) = 0;
  // }
  return TKalmanFilter::predict();
}

const cv::Mat KalmanTracker::get_state(void) {
  return xyah2ltrb(statePost(cv::Rect(0, 0, 1, 4)));
  // return cv::Mat(this->ltrb);
}

cv::Vec2f speed_direction(cv::Vec4f bbox1, cv::Vec4f bbox2) {
  cv::Vec2f center1, center2, speed;
  center1[0] = (bbox1[0]+bbox1[2])/2.0;
  center1[1] = (bbox1[1]+bbox1[3])/2.0;
  center2[0] = (bbox2[0]+bbox2[2])/2.0;
  center2[1] = (bbox2[1]+bbox2[3])/2.0;
  speed[0] = center1[0]-center2[0];
  speed[1] = center1[1]-center2[1];
  float norm = sqrt(speed[0]*speed[0] + speed[1]*speed[1]) + 1e-6;
  speed /= norm;
  return speed;
}

void KalmanTracker::update(cv::Vec4f dets, bool angle_cost) {
  if(angle_cost && this->last_observation[0]>0) {
    cv::Vec4f previous_box(-1,-1,-1,-1);
    for (int i=0; i<this->delta_t; i++) {
      int dt = this->delta_t - i;
      if (this->observations.find(this->age - dt)!=this->observations.end()) {
        previous_box = this->observations[this->age-dt];
        break;
      }
    }
    if (previous_box[0]<0) {
      previous_box = this->last_observation;
    }
    this->velocity = speed_direction(dets, previous_box);
  }

  auto det_xyah = ltrb2xyah(dets);
  TKalmanFilter::correct(cv::Mat(det_xyah));
  this->last_observation = dets;
  this->observations[this->age] = dets;
  this->length += 1;
  this->state = Tracked;
  this->is_activated = true;
  this->time_since_update = 0;
  this->hits += 1;
  this->hit_streak += 1;
}

void KalmanTracker::activate(int timestamp_) {
  TKalmanFilter::init(cv::Mat(xyah));
  length = 0;
  state = Tracked;
  if (timestamp_ == 1) {
    is_activated = true;
  }
  timestamp = timestamp_;
  starttime = timestamp_;
}

void KalmanTracker::reactivate(KalmanTracker *traj, int timestamp_, bool newid) {
  TKalmanFilter::correct(cv::Mat(traj->xyah));
  length = 0;
  state = Tracked;
  is_activated = true;
  timestamp = timestamp_;
  if (newid) alloc_id();
}

