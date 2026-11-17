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

The `i3_ray_serve` package provides a scalable way to easily deploy multiple Ray Serve applications to a single Ray Cluster. Each application defined in the Serve Config manifest is used to deploy an instance of a inference runtime platform, allowing for multiple different runtimes per Serve Deployment on a single ray cluster, or the same inference platform with different inference sessions tuned to different parameters.

This done to abstract the deployment of a model without needing to rewrite Ray Serve code to handle the specifics of a model, so that all that is required to provide IaaS to your model is to wrtie the serve config.

## Configuration

This implementation expects a ServeConfigV2 YAML file for configuring the Ray Serve application and deployments.

```
applications:
  - name: Gateway
    route_prefix: /v2
    import_path: server:build_gateway
    args:
      models:
        - application: OnnxInferenceBackend
          name: classifier
          version: "3"
    runtime_env:
      # Only relevant for running locally; if running via container platform, set env variables through platform specs
      env_vars:
        CI: "True"
        LOG_LEVEL: INFO
  - name: OnnxInferenceBackend
    route_prefix: /v2/models/classifier
    import_path: server:build
    args:
      # runtime should match V2 Platform specs, and have an appropriate InferenceBackend:
      #   https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol#platforms
      runtime: onnx_onnxv1
      model_name: tglauch_classifier
      model_version: "3"
      model_repo: /cvmfs/icecube.opensciencegrid.org/users/briedel/ml/models
      model_path: /home/asheperd/work/models/model.onnx
      batch_max_size: 250
      batch_wait_time_out_s: 0.01
      # Options are InferenceBackend specific configuration
      options:
        providers: [CUDAExecutionProvider, CPUExectionProvider]
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
Each Ray Serve instance needs a `Gateway` application to handle `/v2/health/live` and `/v2` for 
### Required Values
#### application
**name**: unique application name, use something pertaining to the runtime platform
**route_prefix**: route that should prefixed to paths on for this application. Should be `/v2/models/{model_name}`
**import_path**: server:build

#### args
**runtime**: Should be a runtime platform listed in [KServe V2 platforms](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol#platforms)
              currently only supports `onnx_onnxv1`
**model_name**: Model name
**model_version**: Model version. Can be used to differentiate multiple versions for a single model
**model_repository**: path to find the model repository

### Optional Values
#### args
**model_path**: full posix path find your model file, otherwise constructed from model name, version, and repository
**options**: dictionary that is passed to the InferenceBackend for runtime platform specific options
**batch_max_size**: maximum size ray should batch inference requests to
**batch_wait_time_out_s**: how long to wait to fill out batched request


### Environment Variables

**CI**: Should the application use OpenIDConnec for auth? Default = False ()
**LOG_LEVEL**: INFO, CRIT, DEBUG
**AUTH_AUDIENCE**: Client name to use
**AUTH_OPENID_URL**: URL to find `/.well-known` endpoint for `AUTH_AUDIENCE`


## Deploying

Check the `distributions` directory for more info and examples. Currently examples for running locally and a helm chart for deployment to a kubernetes cluster


### Locally

```
uv venv
uv pip install -r pyproject.toml
cd i3_ray_serve
serve run ../distribution/local/serve_config.yaml
``` 

### Helm


```
cd distribution/helm
helm upgrade --install -n i3_ray i3_ray_serve -f /path/to/my/values.yaml . 
```
