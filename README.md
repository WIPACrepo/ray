<!--- Top of README Badges (automated) --->
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/WIPACrepo/ray?include_prereleases)](https://github.com/WIPACrepo/ray) [![GitHub issues](https://img.shields.io/github/issues/WIPACrepo/ray)](https://github.com/WIPACrepo/ray/issues?q=is%3Aissue+sort%3Aupdated-desc+is%3Aopen) [![GitHub pull requests](https://img.shields.io/github/issues-pr/WIPACrepo/ray)](https://github.com/WIPACrepo/ray/pulls?q=is%3Apr+sort%3Aupdated-desc+is%3Aopen)
<!--- End of README Badges (automated) --->
# ray
Interfacing with ray.io
<!--- Top of README Metadata Section (automated) --->

<!--- note: this information is pulled from the pyproject.toml --->

<dl>
    <dt><sub>URLs</sub></dt>
    <dd><sub><a href='https://github.com/WIPACrepo/ray'>Homepage</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href='https://github.com/WIPACrepo/ray/issues'>Tracker</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href='https://github.com/WIPACrepo/ray'>Source</a></sub></dd>
</dl>

<br>
<!--- End of README Metadata Section (automated) --->

Ray Serve is a scalable model serving library for building online inference APIs. This repository contains an implementation of Ray for creating IaaS deloyments via [KServe Inference Protocol V2](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol#httprest-api-payloads)
## Deploying

This implementation expects a ServeConfigV2 YAML file for configuring the Ray Serve application and deployments.

```
applications:
  - name: OnnxInferenceBackend
    route_prefix: /v2/models/classifier
    import_path: server:build
    args:
      # runtime should match V2 Platform specs, and have an appropriate InferenceBackend:
      #   https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol#platforms
      runtime: onnx_onnxv1
      model_name: classifier
      model_version: "3"
      model_repo: /path/to/model
      # Options are InferenceBackend specific configuration
      options:
        ...
    runtime_env:
      # Only relevant for running locally; if running via container platform, set env variables through platform specs
      env_vars:
        CI: "True"
        LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}:${VIRTUAL_ENV}/lib
        LOG_LEVEL: INFO
    deployments:
      # Should be confgiuration passed to deployment handle @serve.deployment
      #   https://docs.ray.io/en/master/serve/configure-serve-deployment.html
      - name: GenericServe
        user_config:
          batch_max_size: 1
          batch_wait_timeout_s: 0.01
        max_ongoing_requests: 2
        max_queued_requests: 10
        backpressure_config:
          status_code: 429
          retry_after_s: 10
        autoscaling_config:
          target_ongoing_requests: 1
          min_replicas: 1
          max_replicas: 1
```
### Required Values
**name**: unique application name, use something pertaining to the runtime platform
**route_prefix**: route that should prefixed to paths on for this application. Should be `/v2/models/{model_name}`
**import_path**: should always be i3_ray_serve:server:build

#### args
**args.runtime**: Should be a runtime platform listed in [KServe V2 platforms](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol#platforms)
              currently only supports `onnx_onnxv1`
**args.model_name**: Model name
**args.model_version**: Model version. Can be used to differentiate multiple versions for a single model
**args.model_repository**: where to find the model

### Optional Values
#### args
**args.model_path**: full posix path find your model file, otherwise constructed from model name, version, and repository
#### 
**options**: dictionary that is passed to the InferenceBackend for runtime platform specific options
