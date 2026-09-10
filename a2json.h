#pragma once
#include <string_view>

// NAM A2 json template
constexpr std::string_view A2JsonData = R"(
{
  "version": "0.7.0",
  "metadata": {
    "date": {{DATE}},
    "loudness": {{LOUDNESS}}
  },
  "sample_rate": 48000.0,
  "architecture": "WaveNet",
  "config": {
    "layers": [
      {
        "input_size": 1,
        "condition_size": 1,
        "head": {
          "out_channels": 1,
          "kernel_size": 16,
          "bias": true
        },
        "channels": {{CHANNELS}},
        "kernel_sizes": [ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6 ],
        "dilations": [ 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239 ],
        "activation": [
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          },
          {
            "type": "LeakyReLU",
            "negative_slope": 0.01
          }
        ],
        "bottleneck": {{CHANNELS}},
        "head1x1": {
          "active": false,
          "out_channels": 1,
          "groups": 1
        },
        "layer1x1": {
          "active": true,
          "groups": 1
        },
        "groups_input": 1,
        "groups_input_mixin": 1,
        "conv_pre_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "conv_post_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "input_mixin_pre_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "input_mixin_post_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "activation_pre_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "activation_post_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "layer1x1_post_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "head1x1_post_film": {
          "active": false,
          "shift": true,
          "groups": 1
        },
        "gating_mode": [ "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "none" ],
        "secondary_activation": [ null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null ],
        "slimmable": null
      }
    ],
    "head": null,
    "head_scale": {{HEADSCALE}}
  },
  "weights": [ {{WEIGHTS}} ]
}
)";
      