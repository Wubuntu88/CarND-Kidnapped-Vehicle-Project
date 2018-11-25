/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized) {
		default_random_engine gen;
		num_particles = 200;

		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		for(unsigned int i = 0; i < num_particles; ++i) {
			Particle particle;
			particle.id = i;
			particle.x = dist_x(gen);
			particle.y = dist_y(gen);
			particle.theta = dist_theta(gen);
			particle.weight = 1.0;

			particles.push_back(particle);
			weights.push_back(particle.weight);
		}
		is_initialized = true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle = particles[i];
		if (fabs(yaw_rate) > 0000.1) {
			particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
		}else {
			particle.x += velocity * cos(particle.theta) * delta_t;
			particle.y += velocity * sin(particle.theta) * delta_t;
		}
		particle.theta += yaw_rate * delta_t;

		normal_distribution<double> dist_x(particle.x, std_pos[0]);
		normal_distribution<double> dist_y(particle.y, std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

		particles[i] = particle;  // put particle back in particles[i] because it is a struct.
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i = 0; i < observations.size(); ++i) {
		LandmarkObs obs = observations[i];
		double minDistance = numeric_limits<double>::max();
		int id = -1;
		for(unsigned int k = 0; k < predicted.size(); ++k) {
			LandmarkObs currentPrediction = predicted[k];
			double currentDistance = dist(obs.x, obs.y, currentPrediction.x, currentPrediction.y);
			if(currentDistance < minDistance) {
				minDistance = currentDistance;
				id = currentPrediction.id;
			}
		}
		observations[i].id = id;
	}
}

LandmarkObs ParticleFilter::transformObservation(Particle particle, LandmarkObs obs) {
	LandmarkObs transformedObservation;
	transformedObservation.id = obs.id;
	transformedObservation.x = cos(particle.theta) * obs.x - sin(particle.theta) * obs.y + particle.x;
	transformedObservation.y = sin(particle.theta) * obs.x + cos(particle.theta) * obs.y + particle.y;
	return transformedObservation;
}

double ParticleFilter::multiVarGaussian(double landmark_x, double landmark_y,
										double std_x, double std_y,
										double obs_x, double obs_y) {
	double coefficient = 1.0 / (2.0 * M_PI * std_x * std_y);
	double exp_pt_1 = pow(landmark_x - obs_x, 2) / (2 * pow(std_x, 2));
	double exp_pt_2 = pow(landmark_y - obs_y, 2) / (2 * pow(std_y, 2));
	return coefficient * exp( - (exp_pt_1 + exp_pt_2));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(unsigned int i = 0; i < num_particles; ++i){
		Particle particle = particles[i];

		vector<LandmarkObs> predictions;

		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			LandmarkObs landmarkObs = LandmarkObs {landmark_id, landmark_x, landmark_y};

			float distance = dist(particle.x, particle.y, landmarkObs.x, landmarkObs.y);
			if( distance < sensor_range) {
				predictions.push_back(landmarkObs);
			}
		}

		vector<LandmarkObs> transformedObservations;
		for(unsigned int c = 0; c < observations.size(); ++c) {
			LandmarkObs transformedObs = transformObservation(particle, observations[c]); // maybe wrong
			transformedObservations.push_back(transformedObs);
		}

		dataAssociation(predictions, transformedObservations);

		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < transformedObservations.size(); ++j) {
			double obs_x = transformedObservations[j].x;
			double obs_y = transformedObservations[j].y;
			double pred_x, pred_y;

			int associated_prediction = transformedObservations[j].id;

			for(int c = 0; c < predictions.size(); ++c) {
				if(predictions[c].id == associated_prediction) {
					pred_x = predictions[c].x;
					pred_y = predictions[c].y;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double obs_w = multiVarGaussian(pred_x, pred_y,
											std_x, std_y,
											obs_x, obs_y);

			particles[i].weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<Particle> resampled_particles;

	vector<double> weights;
	for(int i = 0; i < num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}

	// get random index on resampling wheel
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> unirealdist(0.0, max_weight);
	double beta = 0.0;

	for(unsigned int i = 0; i < num_particles; ++i) {
		beta += 2 * unirealdist(gen);
		while(beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
