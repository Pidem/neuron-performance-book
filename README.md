# From Data Scientist to Performance Engineer

Making the most of your PyTorch models on Neuron chips.



## Build the book and read locally

```bash
pip install -r requirements.txt
jupyter-book build .
open _build/html/index.html
```

## Deploy a trn2 instance

1. Upload the Neuron Explorer extension to S3:
   ```bash
   aws s3 cp neuron-explorer-extension/amazonwebservices.neuron-explorer-2.30.0.vsix \
     s3://cf-templates-pidemal-ap-southeast-4/neuron-explorer-extension/
   ```

2. Deploy the CloudFormation stack:
   ```bash
   AWS_PAGER="" aws cloudformation deploy \
     --template-file workshop-stack.yml \
     --stack-name neuron-workshop \
     --capabilities CAPABILITY_IAM \
     --region ap-southeast-4 \
     --s3-bucket cf-templates-pidemal-ap-southeast-4 \
     --s3-prefix cfn-templates
   ```

3. Get the Code Editor URL:
   ```bash
   aws cloudformation describe-stacks --stack-name neuron-workshop \
     --region ap-southeast-4 --query "Stacks[0].Outputs[?OutputKey=='URL'].OutputValue" --output text
   ```

4. Activate the Python environment in the terminal:
   ```bash
   source /workshop/workspace/native_venv/bin/activate
   ```

## Book Progress
- [X] Intro 
- [X] Part I
- [ ] Part II
- [ ] Part III
- End to end examples with nki.language and nki.isa?
- Repos of existing kernels
- Multi-core/multichip patterns?


## Deploy

Pushes to `main` auto-deploy to GitHub Pages via the workflow in `.github/workflows/deploy.yml`.

Live at: https://pidem.github.io/neuron-performance-book/
