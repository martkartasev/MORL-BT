using UnityEngine;

namespace Env5
{
    public class MORLEnvController : MonoBehaviour
    {
        public Transform agent;
        public Transform trigger;
        public Transform button;
        public Transform goal;
        public GameObject bridgeDown;
        public GameObject bridgeDown2;
        public GameObject bridgeUp;
        public GameObject bridgeUp2;
        public bool randomBridgeZ;

        private float groundY = 0f;
        private float elevatedGroundY = 4f;
        private float width = 40f;
        private float bridgeWidth = 3f;

        float part1 = 0.25f;
        float part2 = 0.25f;
        float part3 = 0.25f;
        float part4 = 0.25f;
        float playerScale = 1f;
        float buttonHeight = 0.0002f;
        float buttonScale = 4f;
        float margin = 1f;

        float x0;
        float x1;
        float x2;
        float x3;
        float x4;
        float height;
        float bridgeZ;

        public float X3 => x3;
        public float X1 => x1;
        public float Width => width;
        public float ElevatedGroundY => elevatedGroundY;
        public float BridgeWidth => bridgeWidth;

        public float PlayerScale
        {
            get => playerScale;
        }

        public float BridgeZ => bridgeZ;

        public Vector3 BridgeEntranceLeft => new(X1, ElevatedGroundY, BridgeZ);
        public Vector3 BridgeEntranceLeft2 => new(X1, ElevatedGroundY, bridgeDown2.transform.localPosition.z);

        void Awake()
        {
            Initialize();
        }

        void FixedUpdate()
        {
            if (Button1Pressed())
            {
                bridgeDown.SetActive(true);
                bridgeDown2.SetActive(true);
                bridgeUp.SetActive(false);
                bridgeUp2.SetActive(false);
            }
            else
            {
                bridgeDown.SetActive(false);
                bridgeDown2.SetActive(false);
                bridgeUp.SetActive(true);
                bridgeUp2.SetActive(true);
            }

            if (AtGoal())
            {
                Debug.Log("You win!");
            }
        }

        public bool Button1Pressed()
        {
            return button.gameObject.GetComponent<CollisionDetector>().Touching(trigger.gameObject);
        }

        public bool AtGoal()
        {
            return agent.gameObject.GetComponent<CollisionDetector>().Touching(goal.gameObject);
        }

        public void Initialize()
        {
            x0 = -Width / 2;
            x1 = x0 + Width * part1;
            x2 = x1 + Width * part2;
            x3 = x2 + Width * part3;
            x4 = x3 + Width * part4;
            height = elevatedGroundY - groundY;

            float minX = x0 + margin + playerScale / 2;
            float maxX = x4 - margin - playerScale / 2;
            float maxXTrigger1 = x3 - margin - playerScale / 2;
            float z0 = -Width / 2;
            float z1 = Width / 2;
            float minZ = z0 + margin + playerScale / 2;
            float maxZ = z1 - margin - playerScale / 2;
            float playerY = elevatedGroundY + playerScale / 2;
            float buttonY = elevatedGroundY + buttonHeight / 2;

            agent.localPosition = new Vector3(-18, playerY, -2);
            agent.gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
            trigger.localPosition = new Vector3(-15, playerY, -2);
            trigger.localRotation = Quaternion.Euler(0, 0, 0);

            button.localPosition = new Vector3(-12, buttonY, -2);

            button.localPosition = new Vector3(Random.Range(x0 + buttonScale / 2, x1 - buttonScale / 2), buttonY, Random.Range(z0 + buttonScale / 2, z1 - buttonScale / 2));
            goal.localPosition = new Vector3(Random.Range(x3 + buttonScale / 2, x4 - buttonScale / 2), buttonY, Random.Range(z0 + buttonScale / 2, z1 - buttonScale / 2));

            agent.localPosition = new Vector3(Random.Range(minX, maxX), playerY, Random.Range(minZ, maxZ));
            trigger.localPosition = new Vector3(Random.Range(minX, maxXTrigger1), playerY, Random.Range(minZ, maxZ));

            bridgeZ = randomBridgeZ ? Random.Range(z0 + bridgeWidth / 2, z1 - bridgeWidth / 2) : 0f;
            var bridgeDownY = 3.95f;
            var bridgeUpY = 11.08f;
            var bridgeUpX = 2.91f;
            bridgeDown.transform.localPosition = new Vector3(x2, bridgeDownY, bridgeZ);
            bridgeUp.transform.localPosition = new Vector3(bridgeUpX, bridgeUpY, bridgeZ);

            bridgeDown.SetActive(false);
            bridgeDown2.SetActive(false);
            bridgeUp.SetActive(true);
            bridgeUp2.SetActive(true);
        }

        public void Reset()
        {
            var playerController = agent.GetComponentInParent<MORLPlayerController>();
            playerController.StopControl();
            Initialize();
        }
    }
}